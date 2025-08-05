"""Auto-scaling and resource management for carbon-aware training."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import math

from .types import CarbonIntensity, TrainingState
from .exceptions import SchedulingError, ConfigurationError


logger = logging.getLogger(__name__)


class ScalingEvent(str, Enum):
    """Types of scaling events."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MIGRATE = "migrate"
    PAUSE = "pause"
    RESUME = "resume"


@dataclass
class ResourceRequirement:
    """Resource requirements for training."""
    cpu_cores: int
    memory_gb: int
    gpu_count: int
    gpu_type: Optional[str] = None
    storage_gb: int = 100
    network_bandwidth_gbps: float = 1.0


@dataclass
class ComputeNode:
    """Compute node configuration."""
    node_id: str
    region: str
    available_cpus: int
    available_memory_gb: int
    available_gpus: int
    gpu_type: str
    cost_per_hour: float
    carbon_intensity: Optional[float] = None
    utilization: float = 0.0
    last_updated: datetime = None


@dataclass
class ScalingDecision:
    """Scaling decision with rationale."""
    event: ScalingEvent
    target_nodes: int
    target_region: Optional[str] = None
    reasoning: str = ""
    estimated_cost_change: float = 0.0
    estimated_carbon_change: float = 0.0
    urgency: float = 0.5  # 0-1 scale


class CarbonAwareAutoScaler:
    """Intelligent auto-scaling based on carbon intensity and performance."""
    
    def __init__(
        self,
        min_nodes: int = 1,
        max_nodes: int = 10,
        target_utilization: float = 0.7,
        carbon_threshold: float = 100.0,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3,
        cooldown_minutes: int = 5
    ):
        """Initialize carbon-aware auto-scaler.
        
        Args:
            min_nodes: Minimum number of nodes
            max_nodes: Maximum number of nodes
            target_utilization: Target resource utilization
            carbon_threshold: Carbon intensity threshold for scaling decisions
            scale_up_threshold: Utilization threshold for scaling up
            scale_down_threshold: Utilization threshold for scaling down
            cooldown_minutes: Cooldown period between scaling events
        """
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.target_utilization = target_utilization
        self.carbon_threshold = carbon_threshold
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_period = timedelta(minutes=cooldown_minutes)
        
        # State tracking
        self.current_nodes: Dict[str, ComputeNode] = {}
        self.scaling_history: List[Tuple[datetime, ScalingDecision]] = []
        self.last_scaling_event: Optional[datetime] = None
        
        # Performance metrics
        self._utilization_history: List[Tuple[datetime, float]] = []
        self._carbon_history: List[Tuple[datetime, float, str]] = []
        
        # Callbacks
        self._scaling_callbacks: List[Callable] = []
    
    def add_node(self, node: ComputeNode) -> None:
        """Add compute node to available pool.
        
        Args:
            node: Compute node to add
        """
        self.current_nodes[node.node_id] = node
        logger.info(f"Added compute node: {node.node_id} in {node.region}")
    
    def remove_node(self, node_id: str) -> bool:
        """Remove compute node from pool.
        
        Args:
            node_id: Node ID to remove
            
        Returns:
            True if node was removed
        """
        if node_id in self.current_nodes:
            node = self.current_nodes[node_id]
            del self.current_nodes[node_id]
            logger.info(f"Removed compute node: {node_id} from {node.region}")
            return True
        return False
    
    def update_node_metrics(
        self,
        node_id: str,
        utilization: float,
        carbon_intensity: Optional[float] = None
    ) -> None:
        """Update node performance metrics.
        
        Args:
            node_id: Node ID
            utilization: Current utilization (0-1)
            carbon_intensity: Current carbon intensity
        """
        if node_id not in self.current_nodes:
            logger.warning(f"Unknown node: {node_id}")
            return
        
        node = self.current_nodes[node_id]
        node.utilization = utilization
        node.last_updated = datetime.now()
        
        if carbon_intensity is not None:
            node.carbon_intensity = carbon_intensity
            self._carbon_history.append((datetime.now(), carbon_intensity, node.region))
        
        # Keep limited history
        if len(self._carbon_history) > 1000:
            self._carbon_history = self._carbon_history[-500:]
    
    async def evaluate_scaling_decision(self) -> Optional[ScalingDecision]:
        """Evaluate whether scaling action is needed.
        
        Returns:
            Scaling decision or None if no action needed
        """
        if not self.current_nodes:
            return None
        
        # Check cooldown period
        if (self.last_scaling_event and 
            datetime.now() - self.last_scaling_event < self.cooldown_period):
            return None
        
        # Calculate current metrics
        avg_utilization = self._calculate_average_utilization()
        avg_carbon_intensity = self._calculate_average_carbon_intensity()
        active_nodes = len([n for n in self.current_nodes.values() if n.utilization > 0.1])
        
        # Record utilization history
        self._utilization_history.append((datetime.now(), avg_utilization))
        if len(self._utilization_history) > 100:
            self._utilization_history.pop(0)
        
        # Evaluate scaling decisions
        decision = None
        
        # High carbon intensity - consider pausing or migrating
        if avg_carbon_intensity > self.carbon_threshold * 1.5:
            decision = await self._evaluate_carbon_mitigation()
        
        # High utilization - scale up
        elif avg_utilization > self.scale_up_threshold and active_nodes < self.max_nodes:
            decision = await self._evaluate_scale_up(avg_utilization, avg_carbon_intensity)
        
        # Low utilization - scale down
        elif avg_utilization < self.scale_down_threshold and active_nodes > self.min_nodes:
            decision = await self._evaluate_scale_down(avg_utilization, avg_carbon_intensity)
        
        # Moderate carbon but opportunity to migrate to cleaner region
        elif avg_carbon_intensity > self.carbon_threshold:
            decision = await self._evaluate_migration()
        
        return decision
    
    async def _evaluate_carbon_mitigation(self) -> Optional[ScalingDecision]:
        """Evaluate carbon mitigation strategies.
        
        Returns:
            Scaling decision for carbon mitigation
        """
        # Check if cleaner regions available
        cleanest_region = self._find_cleanest_region()
        current_avg_carbon = self._calculate_average_carbon_intensity()
        
        if cleanest_region and cleanest_region[1] < current_avg_carbon * 0.7:
            # Significant carbon improvement available through migration
            return ScalingDecision(
                event=ScalingEvent.MIGRATE,
                target_nodes=len(self.current_nodes),
                target_region=cleanest_region[0],
                reasoning=f"Migrate to cleaner region {cleanest_region[0]} "
                         f"({cleanest_region[1]:.1f} vs {current_avg_carbon:.1f} gCO2/kWh)",
                estimated_carbon_change=current_avg_carbon - cleanest_region[1],
                urgency=0.8
            )
        else:
            # No good migration option, consider pausing
            return ScalingDecision(
                event=ScalingEvent.PAUSE,
                target_nodes=len(self.current_nodes),
                reasoning=f"Pause training due to high carbon intensity "
                         f"({current_avg_carbon:.1f} gCO2/kWh)",
                estimated_carbon_change=-current_avg_carbon,
                urgency=0.9
            )
    
    async def _evaluate_scale_up(
        self, 
        utilization: float, 
        carbon_intensity: float
    ) -> Optional[ScalingDecision]:
        """Evaluate scaling up resources.
        
        Args:
            utilization: Current utilization
            carbon_intensity: Current carbon intensity
            
        Returns:
            Scale up decision
        """
        # Calculate optimal number of nodes
        current_nodes = len(self.current_nodes)
        target_nodes = min(
            math.ceil(current_nodes * utilization / self.target_utilization),
            self.max_nodes
        )
        
        if target_nodes <= current_nodes:
            return None
        
        # Consider carbon impact
        carbon_factor = 1.0
        if carbon_intensity > self.carbon_threshold:
            carbon_factor = 0.5  # Be more conservative when carbon is high
        
        target_nodes = max(current_nodes + 1, int(target_nodes * carbon_factor))
        
        # Find best region for new nodes
        best_region = self._find_cleanest_region()
        target_region = best_region[0] if best_region else None
        
        return ScalingDecision(
            event=ScalingEvent.SCALE_UP,
            target_nodes=target_nodes,
            target_region=target_region,
            reasoning=f"Scale up due to high utilization ({utilization:.1%}), "
                     f"target: {target_nodes} nodes",
            estimated_cost_change=self._estimate_cost_change(current_nodes, target_nodes),
            estimated_carbon_change=self._estimate_carbon_change(current_nodes, target_nodes),
            urgency=min(1.0, (utilization - self.scale_up_threshold) * 2)
        )
    
    async def _evaluate_scale_down(
        self, 
        utilization: float, 
        carbon_intensity: float
    ) -> Optional[ScalingDecision]:
        """Evaluate scaling down resources.
        
        Args:
            utilization: Current utilization
            carbon_intensity: Current carbon intensity
            
        Returns:
            Scale down decision
        """
        current_nodes = len(self.current_nodes)
        target_nodes = max(
            math.ceil(current_nodes * utilization / self.target_utilization),
            self.min_nodes
        )
        
        if target_nodes >= current_nodes:
            return None
        
        # Be more aggressive about scaling down when carbon is high
        if carbon_intensity > self.carbon_threshold:
            target_nodes = max(self.min_nodes, target_nodes - 1)
        
        return ScalingDecision(
            event=ScalingEvent.SCALE_DOWN,
            target_nodes=target_nodes,
            reasoning=f"Scale down due to low utilization ({utilization:.1%}), "
                     f"target: {target_nodes} nodes",
            estimated_cost_change=self._estimate_cost_change(current_nodes, target_nodes),
            estimated_carbon_change=self._estimate_carbon_change(current_nodes, target_nodes),
            urgency=0.3
        )
    
    async def _evaluate_migration(self) -> Optional[ScalingDecision]:
        """Evaluate migration to cleaner regions.
        
        Returns:
            Migration decision
        """
        cleanest_region = self._find_cleanest_region()
        current_avg_carbon = self._calculate_average_carbon_intensity()
        
        if not cleanest_region or cleanest_region[1] >= current_avg_carbon * 0.9:
            return None  # No significant improvement available
        
        return ScalingDecision(
            event=ScalingEvent.MIGRATE,
            target_nodes=len(self.current_nodes),
            target_region=cleanest_region[0],
            reasoning=f"Migrate to reduce carbon intensity from "
                     f"{current_avg_carbon:.1f} to {cleanest_region[1]:.1f} gCO2/kWh",
            estimated_carbon_change=current_avg_carbon - cleanest_region[1],
            urgency=0.6
        )
    
    def _calculate_average_utilization(self) -> float:
        """Calculate average utilization across all nodes.
        
        Returns:
            Average utilization (0-1)
        """
        if not self.current_nodes:
            return 0.0
        
        active_nodes = [n for n in self.current_nodes.values() if n.last_updated]
        if not active_nodes:
            return 0.0
        
        return sum(n.utilization for n in active_nodes) / len(active_nodes)
    
    def _calculate_average_carbon_intensity(self) -> float:
        """Calculate average carbon intensity across regions.
        
        Returns:
            Average carbon intensity (gCO2/kWh)
        """
        if not self._carbon_history:
            return self.carbon_threshold  # Default assumption
        
        # Use recent measurements (last hour)
        recent_cutoff = datetime.now() - timedelta(hours=1)
        recent_measurements = [
            intensity for timestamp, intensity, region in self._carbon_history
            if timestamp > recent_cutoff
        ]
        
        if not recent_measurements:
            return self.carbon_threshold
        
        return sum(recent_measurements) / len(recent_measurements)
    
    def _find_cleanest_region(self) -> Optional[Tuple[str, float]]:
        """Find region with lowest carbon intensity.
        
        Returns:
            Tuple of (region, carbon_intensity) or None
        """
        if not self._carbon_history:
            return None
        
        # Get most recent carbon intensity per region
        region_carbon = {}
        for timestamp, intensity, region in reversed(self._carbon_history):
            if region not in region_carbon:
                region_carbon[region] = intensity
        
        if not region_carbon:
            return None
        
        cleanest_region = min(region_carbon.items(), key=lambda x: x[1])
        return cleanest_region
    
    def _estimate_cost_change(self, current_nodes: int, target_nodes: int) -> float:
        """Estimate cost change from scaling.
        
        Args:
            current_nodes: Current number of nodes
            target_nodes: Target number of nodes
            
        Returns:
            Estimated cost change per hour
        """
        if not self.current_nodes:
            return 0.0
        
        avg_cost_per_node = sum(n.cost_per_hour for n in self.current_nodes.values()) / len(self.current_nodes)
        node_change = target_nodes - current_nodes
        
        return node_change * avg_cost_per_node
    
    def _estimate_carbon_change(self, current_nodes: int, target_nodes: int) -> float:
        """Estimate carbon impact change from scaling.
        
        Args:
            current_nodes: Current number of nodes
            target_nodes: Target number of nodes
            
        Returns:
            Estimated carbon change (kg CO2/hour)
        """
        # Simplified calculation: assume 400W per GPU, 150 gCO2/kWh average
        avg_carbon_intensity = self._calculate_average_carbon_intensity()
        power_per_node_kw = 0.4  # 400W GPU
        
        node_change = target_nodes - current_nodes
        carbon_change_kg_per_hour = (node_change * power_per_node_kw * avg_carbon_intensity) / 1000
        
        return carbon_change_kg_per_hour
    
    async def execute_scaling_decision(self, decision: ScalingDecision) -> bool:
        """Execute a scaling decision.
        
        Args:
            decision: Scaling decision to execute
            
        Returns:
            True if execution successful
        """
        try:
            # Record the decision
            self.scaling_history.append((datetime.now(), decision))
            self.last_scaling_event = datetime.now()
            
            # Execute based on decision type
            if decision.event == ScalingEvent.SCALE_UP:
                success = await self._execute_scale_up(decision)
            elif decision.event == ScalingEvent.SCALE_DOWN:
                success = await self._execute_scale_down(decision)
            elif decision.event == ScalingEvent.MIGRATE:
                success = await self._execute_migration(decision)
            elif decision.event == ScalingEvent.PAUSE:
                success = await self._execute_pause(decision)
            elif decision.event == ScalingEvent.RESUME:
                success = await self._execute_resume(decision)
            else:
                logger.warning(f"Unknown scaling event: {decision.event}")
                return False
            
            if success:
                logger.info(f"Executed scaling decision: {decision.reasoning}")
                
                # Notify callbacks
                for callback in self._scaling_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(decision)
                        else:
                            callback(decision)
                    except Exception as e:
                        logger.error(f"Scaling callback error: {e}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to execute scaling decision: {e}")
            return False
    
    async def _execute_scale_up(self, decision: ScalingDecision) -> bool:
        """Execute scale up decision.
        
        Args:
            decision: Scale up decision
            
        Returns:
            True if successful
        """
        current_count = len(self.current_nodes)
        target_count = decision.target_nodes
        nodes_to_add = target_count - current_count
        
        logger.info(f"Scaling up: adding {nodes_to_add} nodes")
        
        # This would integrate with actual cloud provider APIs
        # For now, simulate by adding placeholder nodes
        for i in range(nodes_to_add):
            node_id = f"scaled_node_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
            region = decision.target_region or "US-CA"
            
            node = ComputeNode(
                node_id=node_id,
                region=region,
                available_cpus=8,
                available_memory_gb=32,
                available_gpus=1,
                gpu_type="V100",
                cost_per_hour=2.5,
                utilization=0.0,
                last_updated=datetime.now()
            )
            
            self.add_node(node)
        
        return True
    
    async def _execute_scale_down(self, decision: ScalingDecision) -> bool:
        """Execute scale down decision.
        
        Args:
            decision: Scale down decision
            
        Returns:
            True if successful
        """
        current_count = len(self.current_nodes)
        target_count = decision.target_nodes
        nodes_to_remove = current_count - target_count
        
        logger.info(f"Scaling down: removing {nodes_to_remove} nodes")
        
        # Remove nodes with lowest utilization first
        nodes_by_utilization = sorted(
            self.current_nodes.values(),
            key=lambda n: n.utilization
        )
        
        for i in range(min(nodes_to_remove, len(nodes_by_utilization))):
            node = nodes_by_utilization[i]
            self.remove_node(node.node_id)
        
        return True
    
    async def _execute_migration(self, decision: ScalingDecision) -> bool:
        """Execute migration decision.
        
        Args:
            decision: Migration decision
            
        Returns:
            True if successful
        """
        target_region = decision.target_region
        if not target_region:
            return False
        
        logger.info(f"Migrating to region: {target_region}")
        
        # Update all nodes to target region (simplified)
        for node in self.current_nodes.values():
            node.region = target_region
        
        return True
    
    async def _execute_pause(self, decision: ScalingDecision) -> bool:
        """Execute pause decision.
        
        Args:
            decision: Pause decision
            
        Returns:
            True if successful
        """
        logger.info("Pausing training due to high carbon intensity")
        
        # Set all nodes to zero utilization (simplified)
        for node in self.current_nodes.values():
            node.utilization = 0.0
        
        return True
    
    async def _execute_resume(self, decision: ScalingDecision) -> bool:
        """Execute resume decision.
        
        Args:
            decision: Resume decision
            
        Returns:
            True if successful
        """
        logger.info("Resuming training")
        
        # This would restore normal operation
        return True
    
    def add_scaling_callback(self, callback: Callable) -> None:
        """Add callback for scaling events.
        
        Args:
            callback: Callback function
        """
        self._scaling_callbacks.append(callback)
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics.
        
        Returns:
            Dictionary with scaling statistics
        """
        recent_decisions = [
            decision for timestamp, decision in self.scaling_history
            if timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        decision_counts = {}
        for _, decision in recent_decisions:
            decision_counts[decision.event.value] = decision_counts.get(decision.event.value, 0) + 1
        
        return {
            "current_nodes": len(self.current_nodes),
            "min_nodes": self.min_nodes,
            "max_nodes": self.max_nodes,
            "average_utilization": self._calculate_average_utilization(),
            "average_carbon_intensity": self._calculate_average_carbon_intensity(),
            "last_scaling_event": self.last_scaling_event.isoformat() if self.last_scaling_event else None,
            "scaling_decisions_24h": len(recent_decisions),
            "decision_breakdown": decision_counts,
            "total_cost_per_hour": sum(n.cost_per_hour for n in self.current_nodes.values())
        }