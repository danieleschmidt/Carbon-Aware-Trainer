"""Hyperscale optimization engine for massive deployments."""

import asyncio
import logging
import time
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import concurrent.futures

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from .types import CarbonIntensity, TrainingState
from .enhanced_monitoring import EnhancedMonitoringSystem, MetricType

logger = logging.getLogger(__name__)


class ScalingStrategy(str, Enum):
    """Auto-scaling strategies."""
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    HYBRID = "hybrid"
    CARBON_AWARE = "carbon_aware"


class LoadBalancingMode(str, Enum):
    """Load balancing algorithms."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    CARBON_OPTIMIZED = "carbon_optimized"
    PERFORMANCE_BASED = "performance_based"


@dataclass
class ScalingMetrics:
    """Scaling performance metrics."""
    current_instances: int = 0
    target_instances: int = 0
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    request_rate: float = 0.0
    response_time_p95: float = 0.0
    carbon_intensity: float = 0.0
    cost_per_hour: float = 0.0
    efficiency_score: float = 0.0


@dataclass
class WorkerNode:
    """Represents a worker node in the cluster."""
    node_id: str
    region: str
    cpu_cores: int
    memory_gb: float
    current_load: float = 0.0
    carbon_intensity: float = 0.0
    is_available: bool = True
    last_heartbeat: datetime = field(default_factory=datetime.now)
    performance_score: float = 1.0
    cost_per_hour: float = 0.0
    specializations: Set[str] = field(default_factory=set)


class HyperscaleOptimizer:
    """Advanced optimization engine for hyperscale deployments."""
    
    def __init__(
        self,
        scaling_strategy: ScalingStrategy = ScalingStrategy.HYBRID,
        load_balancing_mode: LoadBalancingMode = LoadBalancingMode.CARBON_OPTIMIZED,
        target_cpu_utilization: float = 70.0,
        target_memory_utilization: float = 80.0,
        scale_up_threshold: float = 80.0,
        scale_down_threshold: float = 40.0,
        min_instances: int = 2,
        max_instances: int = 1000,
        carbon_weight: float = 0.3,
        cost_weight: float = 0.2,
        performance_weight: float = 0.5
    ):
        """Initialize hyperscale optimizer.
        
        Args:
            scaling_strategy: Auto-scaling strategy to use
            load_balancing_mode: Load balancing algorithm
            target_cpu_utilization: Target CPU utilization percentage
            target_memory_utilization: Target memory utilization percentage
            scale_up_threshold: CPU threshold for scaling up
            scale_down_threshold: CPU threshold for scaling down
            min_instances: Minimum number of instances
            max_instances: Maximum number of instances
            carbon_weight: Weight for carbon optimization (0-1)
            cost_weight: Weight for cost optimization (0-1)
            performance_weight: Weight for performance optimization (0-1)
        """
        self.scaling_strategy = scaling_strategy
        self.load_balancing_mode = load_balancing_mode
        self.target_cpu_utilization = target_cpu_utilization
        self.target_memory_utilization = target_memory_utilization
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.min_instances = min_instances
        self.max_instances = max_instances
        
        # Optimization weights (should sum to 1.0)
        total_weight = carbon_weight + cost_weight + performance_weight
        self.carbon_weight = carbon_weight / total_weight
        self.cost_weight = cost_weight / total_weight
        self.performance_weight = performance_weight / total_weight
        
        # Worker node management
        self._worker_nodes: Dict[str, WorkerNode] = {}
        self._node_lock = threading.RLock()
        
        # Load balancing state
        self._round_robin_index = 0
        self._connection_counts: Dict[str, int] = defaultdict(int)
        
        # Scaling metrics and history
        self._scaling_metrics = ScalingMetrics()
        self._metrics_history: deque = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        self._request_history: deque = deque(maxlen=300)   # 5 minutes of request data
        
        # Predictive scaling
        self._load_predictions: deque = deque(maxlen=60)   # 1 hour of predictions
        self._scaling_decisions: deque = deque(maxlen=100) # Recent scaling decisions
        
        # Background tasks
        self._optimizer_task: Optional[asyncio.Task] = None
        self._prediction_task: Optional[asyncio.Task] = None
        self._monitoring_system: Optional[EnhancedMonitoringSystem] = None
        
        # Thread pool for CPU-intensive operations
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        # Callbacks
        self._scaling_callbacks: List[Callable] = []
        self._load_balancing_callbacks: List[Callable] = []
        
        logger.info(f"Hyperscale optimizer initialized with {scaling_strategy.value} scaling")
    
    async def start_optimization(
        self,
        monitoring_system: Optional[EnhancedMonitoringSystem] = None
    ) -> None:
        """Start hyperscale optimization.
        
        Args:
            monitoring_system: Optional monitoring system integration
        """
        if self._optimizer_task and not self._optimizer_task.done():
            return
        
        self._monitoring_system = monitoring_system
        
        # Start optimization tasks
        self._optimizer_task = asyncio.create_task(self._optimization_loop())
        
        if self.scaling_strategy in [ScalingStrategy.PREDICTIVE, ScalingStrategy.HYBRID]:
            self._prediction_task = asyncio.create_task(self._prediction_loop())
        
        logger.info("Started hyperscale optimization engine")
    
    async def stop_optimization(self) -> None:
        """Stop hyperscale optimization."""
        # Cancel tasks
        for task in [self._optimizer_task, self._prediction_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Shutdown thread pool
        self._thread_pool.shutdown(wait=True)
        
        logger.info("Stopped hyperscale optimization engine")
    
    async def _optimization_loop(self) -> None:
        """Main optimization loop."""
        while True:
            try:
                # Update scaling metrics
                await self._update_scaling_metrics()
                
                # Make scaling decisions
                await self._make_scaling_decisions()
                
                # Optimize load balancing
                await self._optimize_load_balancing()
                
                # Update worker health
                await self._update_worker_health()
                
                # Record metrics
                await self._record_optimization_metrics()
                
                await asyncio.sleep(30)  # Optimize every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(30)
    
    async def _prediction_loop(self) -> None:
        """Predictive scaling loop."""
        while True:
            try:
                # Generate load predictions
                await self._generate_load_predictions()
                
                # Apply predictive scaling
                if self.scaling_strategy == ScalingStrategy.PREDICTIVE:
                    await self._apply_predictive_scaling()
                
                await asyncio.sleep(300)  # Predict every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Prediction loop error: {e}")
                await asyncio.sleep(300)
    
    async def _update_scaling_metrics(self) -> None:
        """Update current scaling metrics."""
        try:
            with self._node_lock:
                active_nodes = [node for node in self._worker_nodes.values() if node.is_available]
                
                self._scaling_metrics.current_instances = len(active_nodes)
                
                if active_nodes:
                    # Calculate aggregate metrics
                    total_cpu = sum(node.current_load for node in active_nodes)
                    self._scaling_metrics.cpu_utilization = total_cpu / len(active_nodes)
                    
                    # Estimate memory utilization (simplified)
                    self._scaling_metrics.memory_utilization = min(
                        self._scaling_metrics.cpu_utilization * 1.2, 100.0
                    )
                    
                    # Average carbon intensity
                    self._scaling_metrics.carbon_intensity = sum(
                        node.carbon_intensity for node in active_nodes
                    ) / len(active_nodes)
                    
                    # Total cost
                    self._scaling_metrics.cost_per_hour = sum(
                        node.cost_per_hour for node in active_nodes
                    )
                    
                    # Calculate efficiency score
                    self._scaling_metrics.efficiency_score = self._calculate_efficiency_score()
            
            # Calculate request rate from recent history
            if len(self._request_history) > 1:
                recent_requests = len([r for r in self._request_history if r > time.time() - 60])
                self._scaling_metrics.request_rate = recent_requests / 60.0  # requests per second
            
            # Store metrics history
            self._metrics_history.append({
                'timestamp': datetime.now(),
                'metrics': self._scaling_metrics
            })
            
        except Exception as e:
            logger.error(f"Error updating scaling metrics: {e}")
    
    def _calculate_efficiency_score(self) -> float:
        """Calculate overall system efficiency score (0-1)."""
        try:
            # Performance score (inverse of utilization deviation from target)
            cpu_deviation = abs(self._scaling_metrics.cpu_utilization - self.target_cpu_utilization)
            performance_score = max(0, 1 - (cpu_deviation / 100))
            
            # Carbon score (inverse of carbon intensity, normalized)
            carbon_score = max(0, 1 - (self._scaling_metrics.carbon_intensity / 1000))
            
            # Cost score (simplified - inverse of cost per request)
            cost_per_request = self._scaling_metrics.cost_per_hour / max(1, self._scaling_metrics.request_rate * 3600)
            cost_score = max(0, 1 - (cost_per_request / 0.1))  # Normalize to $0.1 per request
            
            # Weighted combination
            efficiency = (
                self.performance_weight * performance_score +
                self.carbon_weight * carbon_score +
                self.cost_weight * cost_score
            )
            
            return min(1.0, max(0.0, efficiency))
            
        except Exception:
            return 0.5  # Default neutral score
    
    async def _make_scaling_decisions(self) -> None:
        """Make intelligent scaling decisions."""
        try:
            current_instances = self._scaling_metrics.current_instances
            current_cpu = self._scaling_metrics.cpu_utilization
            
            # Determine target instances based on strategy
            if self.scaling_strategy == ScalingStrategy.REACTIVE:
                target_instances = self._calculate_reactive_scaling(current_cpu)
            elif self.scaling_strategy == ScalingStrategy.PREDICTIVE:
                target_instances = await self._calculate_predictive_scaling()
            else:  # HYBRID or CARBON_AWARE
                reactive_target = self._calculate_reactive_scaling(current_cpu)
                predictive_target = await self._calculate_predictive_scaling()
                target_instances = await self._calculate_hybrid_scaling(reactive_target, predictive_target)
            
            # Apply constraints
            target_instances = max(self.min_instances, min(self.max_instances, target_instances))
            
            self._scaling_metrics.target_instances = target_instances
            
            # Execute scaling if needed
            if target_instances != current_instances:
                await self._execute_scaling(target_instances)
            
        except Exception as e:
            logger.error(f"Error making scaling decisions: {e}")
    
    def _calculate_reactive_scaling(self, current_cpu: float) -> int:
        """Calculate reactive scaling based on current metrics."""
        current_instances = self._scaling_metrics.current_instances
        
        if current_cpu > self.scale_up_threshold:
            # Scale up: calculate instances needed for target utilization
            scale_factor = current_cpu / self.target_cpu_utilization
            return math.ceil(current_instances * scale_factor)
        
        elif current_cpu < self.scale_down_threshold and current_instances > self.min_instances:
            # Scale down: but conservatively
            scale_factor = current_cpu / self.target_cpu_utilization
            return max(self.min_instances, math.ceil(current_instances * scale_factor))
        
        return current_instances
    
    async def _calculate_predictive_scaling(self) -> int:
        """Calculate predictive scaling based on forecasts."""
        try:
            if not self._load_predictions:
                return self._scaling_metrics.current_instances
            
            # Use next 15 minutes of predictions
            future_load = list(self._load_predictions)[-15:]  # Next 15 predictions
            
            if not future_load:
                return self._scaling_metrics.current_instances
            
            # Calculate required instances for predicted load
            max_predicted_load = max(future_load)
            target_instances = math.ceil(
                (max_predicted_load / 100) * self._scaling_metrics.current_instances * 1.2  # 20% buffer
            )
            
            return target_instances
            
        except Exception as e:
            logger.error(f"Error in predictive scaling calculation: {e}")
            return self._scaling_metrics.current_instances
    
    async def _calculate_hybrid_scaling(self, reactive_target: int, predictive_target: int) -> int:
        """Calculate hybrid scaling combining reactive and predictive."""
        try:
            # Weight reactive and predictive based on confidence
            reactive_weight = 0.6
            predictive_weight = 0.4
            
            # If carbon-aware, adjust for carbon intensity
            if self.scaling_strategy == ScalingStrategy.CARBON_AWARE:
                carbon_factor = await self._get_carbon_scaling_factor()
                predictive_target = int(predictive_target * carbon_factor)
            
            # Combine targets
            hybrid_target = (
                reactive_weight * reactive_target +
                predictive_weight * predictive_target
            )
            
            return int(hybrid_target)
            
        except Exception as e:
            logger.error(f"Error in hybrid scaling calculation: {e}")
            return reactive_target  # Fall back to reactive
    
    async def _get_carbon_scaling_factor(self) -> float:
        """Get carbon-aware scaling factor (0.5-1.5)."""
        try:
            current_carbon = self._scaling_metrics.carbon_intensity
            
            # Scale down during high carbon periods, up during low carbon
            if current_carbon > 400:  # High carbon
                return 0.7  # Scale down 30%
            elif current_carbon < 100:  # Low carbon
                return 1.3  # Scale up 30%
            else:
                return 1.0  # No adjustment
                
        except Exception:
            return 1.0
    
    async def _execute_scaling(self, target_instances: int) -> None:
        """Execute scaling decision."""
        try:
            current_instances = self._scaling_metrics.current_instances
            
            if target_instances > current_instances:
                # Scale up
                instances_to_add = target_instances - current_instances
                await self._scale_up(instances_to_add)
            
            elif target_instances < current_instances:
                # Scale down
                instances_to_remove = current_instances - target_instances
                await self._scale_down(instances_to_remove)
            
            # Record scaling decision
            self._scaling_decisions.append({
                'timestamp': datetime.now(),
                'from_instances': current_instances,
                'to_instances': target_instances,
                'cpu_utilization': self._scaling_metrics.cpu_utilization,
                'strategy': self.scaling_strategy.value
            })
            
            # Notify callbacks
            for callback in self._scaling_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback('scaling_executed', {
                            'from': current_instances,
                            'to': target_instances,
                            'strategy': self.scaling_strategy.value
                        })
                    else:
                        callback('scaling_executed', {
                            'from': current_instances,
                            'to': target_instances,
                            'strategy': self.scaling_strategy.value
                        })
                except Exception as e:
                    logger.error(f"Scaling callback error: {e}")
            
        except Exception as e:
            logger.error(f"Error executing scaling: {e}")
    
    async def _scale_up(self, instances_to_add: int) -> None:
        """Scale up by adding worker instances."""
        logger.info(f"Scaling up: adding {instances_to_add} instances")
        
        # This would integrate with actual infrastructure APIs
        # For now, simulate by creating worker nodes
        for i in range(instances_to_add):
            node_id = f"worker_{int(time.time())}_{i}"
            
            # Select best region for new instance
            best_region = await self._select_optimal_region()
            
            new_node = WorkerNode(
                node_id=node_id,
                region=best_region,
                cpu_cores=4,  # Default configuration
                memory_gb=16.0,
                current_load=0.0,
                carbon_intensity=await self._get_region_carbon_intensity(best_region),
                cost_per_hour=0.5  # Example cost
            )
            
            await self.register_worker_node(new_node)
    
    async def _scale_down(self, instances_to_remove: int) -> None:
        """Scale down by removing worker instances."""
        logger.info(f"Scaling down: removing {instances_to_remove} instances")
        
        # Select worst performing nodes for removal
        nodes_to_remove = await self._select_nodes_for_removal(instances_to_remove)
        
        for node in nodes_to_remove:
            # Gracefully drain node before removal
            await self._drain_worker_node(node.node_id)
            await self.unregister_worker_node(node.node_id)
    
    async def _select_optimal_region(self) -> str:
        """Select optimal region for new instances."""
        try:
            # This would integrate with carbon APIs and cost data
            # For now, return a default region
            return "us-west-2"  # Low carbon region
        except Exception:
            return "us-east-1"  # Fallback
    
    async def _get_region_carbon_intensity(self, region: str) -> float:
        """Get carbon intensity for a region."""
        try:
            # This would integrate with carbon intensity APIs
            # Return simulated values based on region
            carbon_map = {
                "us-west-2": 50.0,   # Low carbon (hydro)
                "us-east-1": 200.0,  # Medium carbon
                "us-central": 400.0  # High carbon (coal)
            }
            return carbon_map.get(region, 200.0)
        except Exception:
            return 200.0  # Default value
    
    async def _select_nodes_for_removal(self, count: int) -> List[WorkerNode]:
        """Select worker nodes for removal based on efficiency."""
        try:
            with self._node_lock:
                available_nodes = [
                    node for node in self._worker_nodes.values() 
                    if node.is_available
                ]
            
            # Sort by removal priority (lowest efficiency first)
            available_nodes.sort(key=lambda n: (
                n.performance_score,  # Remove low performance nodes first
                -n.carbon_intensity,  # Remove high carbon nodes first
                -n.current_load      # Remove lightly loaded nodes first
            ))
            
            return available_nodes[:count]
            
        except Exception as e:
            logger.error(f"Error selecting nodes for removal: {e}")
            return []
    
    async def _optimize_load_balancing(self) -> None:
        """Optimize load balancing across worker nodes."""
        try:
            if self.load_balancing_mode == LoadBalancingMode.CARBON_OPTIMIZED:
                await self._update_carbon_optimized_weights()
            elif self.load_balancing_mode == LoadBalancingMode.PERFORMANCE_BASED:
                await self._update_performance_weights()
            
        except Exception as e:
            logger.error(f"Error optimizing load balancing: {e}")
    
    async def _update_carbon_optimized_weights(self) -> None:
        """Update node weights for carbon-optimized load balancing."""
        try:
            with self._node_lock:
                for node in self._worker_nodes.values():
                    if node.is_available:
                        # Lower carbon intensity = higher weight
                        carbon_score = max(0.1, 1.0 - (node.carbon_intensity / 1000))
                        load_score = max(0.1, 1.0 - (node.current_load / 100))
                        
                        # Combined score
                        node.performance_score = (carbon_score * 0.6) + (load_score * 0.4)
            
        except Exception as e:
            logger.error(f"Error updating carbon weights: {e}")
    
    async def _update_performance_weights(self) -> None:
        """Update node weights for performance-based load balancing."""
        try:
            with self._node_lock:
                for node in self._worker_nodes.values():
                    if node.is_available:
                        # Performance score based on load and response time
                        load_score = max(0.1, 1.0 - (node.current_load / 100))
                        
                        # Simple performance score (would be more sophisticated in reality)
                        node.performance_score = load_score
            
        except Exception as e:
            logger.error(f"Error updating performance weights: {e}")
    
    async def _generate_load_predictions(self) -> None:
        """Generate load predictions using simple time series analysis."""
        try:
            if not HAS_NUMPY or len(self._metrics_history) < 30:
                return
            
            # Extract CPU utilization history
            cpu_history = [
                entry['metrics'].cpu_utilization 
                for entry in list(self._metrics_history)[-60:]  # Last hour
            ]
            
            if len(cpu_history) < 10:
                return
            
            # Simple moving average prediction
            window_size = min(10, len(cpu_history))
            recent_avg = np.mean(cpu_history[-window_size:])
            trend = np.mean(np.diff(cpu_history[-window_size:]))
            
            # Generate next 15 predictions (15 minutes)
            predictions = []
            for i in range(15):
                predicted_load = recent_avg + (trend * i)
                predicted_load = max(0, min(100, predicted_load))  # Clamp to 0-100%
                predictions.append(predicted_load)
            
            # Store predictions
            self._load_predictions.extend(predictions)
            
        except Exception as e:
            logger.error(f"Error generating load predictions: {e}")
    
    async def _update_worker_health(self) -> None:
        """Update worker node health status."""
        try:
            current_time = datetime.now()
            heartbeat_timeout = timedelta(minutes=5)
            
            with self._node_lock:
                for node in self._worker_nodes.values():
                    # Mark nodes as unavailable if they haven't sent heartbeat
                    if current_time - node.last_heartbeat > heartbeat_timeout:
                        if node.is_available:
                            node.is_available = False
                            logger.warning(f"Worker node {node.node_id} marked as unavailable")
            
        except Exception as e:
            logger.error(f"Error updating worker health: {e}")
    
    async def _record_optimization_metrics(self) -> None:
        """Record optimization metrics to monitoring system."""
        if not self._monitoring_system:
            return
        
        try:
            # Record scaling metrics
            await self._monitoring_system.record_metric(
                "scaling_current_instances",
                self._scaling_metrics.current_instances,
                "count",
                MetricType.SYSTEM
            )
            
            await self._monitoring_system.record_metric(
                "scaling_cpu_utilization",
                self._scaling_metrics.cpu_utilization,
                "percent",
                MetricType.SYSTEM
            )
            
            await self._monitoring_system.record_metric(
                "scaling_efficiency_score",
                self._scaling_metrics.efficiency_score,
                "score",
                MetricType.PERFORMANCE
            )
            
            await self._monitoring_system.record_metric(
                "scaling_carbon_intensity",
                self._scaling_metrics.carbon_intensity,
                "gCO2/kWh",
                MetricType.CARBON
            )
            
            await self._monitoring_system.record_metric(
                "scaling_cost_per_hour",
                self._scaling_metrics.cost_per_hour,
                "USD",
                MetricType.BUSINESS
            )
            
        except Exception as e:
            logger.error(f"Error recording optimization metrics: {e}")
    
    async def register_worker_node(self, node: WorkerNode) -> None:
        """Register a new worker node.
        
        Args:
            node: Worker node to register
        """
        with self._node_lock:
            self._worker_nodes[node.node_id] = node
        
        logger.info(f"Registered worker node {node.node_id} in region {node.region}")
    
    async def unregister_worker_node(self, node_id: str) -> None:
        """Unregister a worker node.
        
        Args:
            node_id: ID of node to unregister
        """
        with self._node_lock:
            if node_id in self._worker_nodes:
                del self._worker_nodes[node_id]
        
        logger.info(f"Unregistered worker node {node_id}")
    
    async def update_worker_metrics(
        self, 
        node_id: str, 
        cpu_load: float, 
        carbon_intensity: Optional[float] = None
    ) -> None:
        """Update worker node metrics.
        
        Args:
            node_id: Worker node ID
            cpu_load: Current CPU load percentage
            carbon_intensity: Optional carbon intensity update
        """
        with self._node_lock:
            if node_id in self._worker_nodes:
                node = self._worker_nodes[node_id]
                node.current_load = cpu_load
                node.last_heartbeat = datetime.now()
                node.is_available = True
                
                if carbon_intensity is not None:
                    node.carbon_intensity = carbon_intensity
    
    async def _drain_worker_node(self, node_id: str) -> None:
        """Gracefully drain a worker node.
        
        Args:
            node_id: Worker node to drain
        """
        logger.info(f"Draining worker node {node_id}")
        
        # This would implement graceful connection draining
        # For now, just mark as unavailable
        with self._node_lock:
            if node_id in self._worker_nodes:
                self._worker_nodes[node_id].is_available = False
    
    def select_worker_node(self, request_context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Select optimal worker node for request.
        
        Args:
            request_context: Optional context for routing decision
            
        Returns:
            Selected worker node ID or None if no nodes available
        """
        try:
            with self._node_lock:
                available_nodes = [
                    node for node in self._worker_nodes.values() 
                    if node.is_available
                ]
            
            if not available_nodes:
                return None
            
            # Apply load balancing algorithm
            if self.load_balancing_mode == LoadBalancingMode.ROUND_ROBIN:
                return self._select_round_robin(available_nodes)
            elif self.load_balancing_mode == LoadBalancingMode.LEAST_CONNECTIONS:
                return self._select_least_connections(available_nodes)
            elif self.load_balancing_mode == LoadBalancingMode.CARBON_OPTIMIZED:
                return self._select_carbon_optimized(available_nodes)
            elif self.load_balancing_mode == LoadBalancingMode.PERFORMANCE_BASED:
                return self._select_performance_based(available_nodes)
            else:  # WEIGHTED_ROUND_ROBIN
                return self._select_weighted_round_robin(available_nodes)
            
        except Exception as e:
            logger.error(f"Error selecting worker node: {e}")
            return None
    
    def _select_round_robin(self, nodes: List[WorkerNode]) -> str:
        """Select node using round-robin."""
        if not nodes:
            return None
        
        self._round_robin_index = (self._round_robin_index + 1) % len(nodes)
        return nodes[self._round_robin_index].node_id
    
    def _select_least_connections(self, nodes: List[WorkerNode]) -> str:
        """Select node with least connections."""
        if not nodes:
            return None
        
        # Find node with minimum connection count
        min_connections = min(self._connection_counts.get(node.node_id, 0) for node in nodes)
        candidates = [
            node for node in nodes 
            if self._connection_counts.get(node.node_id, 0) == min_connections
        ]
        
        # Break ties with performance score
        best_node = max(candidates, key=lambda n: n.performance_score)
        return best_node.node_id
    
    def _select_carbon_optimized(self, nodes: List[WorkerNode]) -> str:
        """Select node optimized for carbon efficiency."""
        if not nodes:
            return None
        
        # Weight by inverse carbon intensity and current load
        def carbon_score(node):
            carbon_weight = max(0.1, 1.0 - (node.carbon_intensity / 1000))
            load_weight = max(0.1, 1.0 - (node.current_load / 100))
            return carbon_weight * 0.7 + load_weight * 0.3
        
        best_node = max(nodes, key=carbon_score)
        return best_node.node_id
    
    def _select_performance_based(self, nodes: List[WorkerNode]) -> str:
        """Select node based on performance score."""
        if not nodes:
            return None
        
        best_node = max(nodes, key=lambda n: n.performance_score)
        return best_node.node_id
    
    def _select_weighted_round_robin(self, nodes: List[WorkerNode]) -> str:
        """Select node using weighted round-robin based on performance."""
        if not nodes:
            return None
        
        # Simple weighted selection based on performance scores
        total_weight = sum(node.performance_score for node in nodes)
        if total_weight == 0:
            return self._select_round_robin(nodes)
        
        import random
        target = random.uniform(0, total_weight)
        
        cumulative = 0
        for node in nodes:
            cumulative += node.performance_score
            if cumulative >= target:
                return node.node_id
        
        return nodes[-1].node_id  # Fallback
    
    def record_request(self, node_id: str, success: bool, response_time_ms: float) -> None:
        """Record request metrics for a worker node.
        
        Args:
            node_id: Worker node that handled the request
            success: Whether request was successful
            response_time_ms: Response time in milliseconds
        """
        # Update connection count
        if success:
            self._connection_counts[node_id] = self._connection_counts.get(node_id, 0) + 1
        
        # Record request in history
        self._request_history.append(time.time())
        
        # Update node performance score based on response time
        with self._node_lock:
            if node_id in self._worker_nodes:
                node = self._worker_nodes[node_id]
                
                # Simple performance score update
                if response_time_ms < 100:  # Fast response
                    node.performance_score = min(1.0, node.performance_score + 0.01)
                elif response_time_ms > 1000:  # Slow response
                    node.performance_score = max(0.1, node.performance_score - 0.05)
    
    def get_scaling_metrics(self) -> Dict[str, Any]:
        """Get current scaling metrics.
        
        Returns:
            Dictionary of scaling metrics
        """
        return {
            "strategy": self.scaling_strategy.value,
            "load_balancing": self.load_balancing_mode.value,
            "current_instances": self._scaling_metrics.current_instances,
            "target_instances": self._scaling_metrics.target_instances,
            "cpu_utilization": self._scaling_metrics.cpu_utilization,
            "memory_utilization": self._scaling_metrics.memory_utilization,
            "request_rate": self._scaling_metrics.request_rate,
            "carbon_intensity": self._scaling_metrics.carbon_intensity,
            "cost_per_hour": self._scaling_metrics.cost_per_hour,
            "efficiency_score": self._scaling_metrics.efficiency_score,
            "worker_nodes": len(self._worker_nodes),
            "available_nodes": len([n for n in self._worker_nodes.values() if n.is_available])
        }
    
    def get_worker_node_status(self) -> List[Dict[str, Any]]:
        """Get status of all worker nodes.
        
        Returns:
            List of worker node status information
        """
        with self._node_lock:
            return [
                {
                    "node_id": node.node_id,
                    "region": node.region,
                    "cpu_cores": node.cpu_cores,
                    "memory_gb": node.memory_gb,
                    "current_load": node.current_load,
                    "carbon_intensity": node.carbon_intensity,
                    "is_available": node.is_available,
                    "performance_score": node.performance_score,
                    "cost_per_hour": node.cost_per_hour,
                    "last_heartbeat": node.last_heartbeat.isoformat()
                }
                for node in self._worker_nodes.values()
            ]
    
    def add_scaling_callback(self, callback: Callable) -> None:
        """Add scaling event callback.
        
        Args:
            callback: Function to call on scaling events
        """
        self._scaling_callbacks.append(callback)
    
    def add_load_balancing_callback(self, callback: Callable) -> None:
        """Add load balancing event callback.
        
        Args:
            callback: Function to call on load balancing events
        """
        self._load_balancing_callbacks.append(callback)