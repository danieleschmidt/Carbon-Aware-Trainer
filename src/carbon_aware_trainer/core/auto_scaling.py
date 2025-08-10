"""Advanced auto-scaling and optimization for carbon-aware training."""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import threading

from .types import CarbonIntensity, TrainingMetrics, TrainingState
from .exceptions import CarbonDataError


logger = logging.getLogger(__name__)


class ScalingStrategy(str, Enum):
    """Auto-scaling strategies."""
    CARBON_AWARE = "carbon_aware"
    PERFORMANCE_BASED = "performance_based"  
    COST_OPTIMIZED = "cost_optimized"
    HYBRID = "hybrid"


class ResourceType(str, Enum):
    """Types of scalable resources."""
    COMPUTE_INSTANCES = "compute_instances"
    GPU_DEVICES = "gpu_devices"
    MEMORY_ALLOCATION = "memory_allocation"
    BATCH_SIZE = "batch_size"
    LEARNING_RATE = "learning_rate"
    PARALLELISM = "parallelism"


@dataclass
class ScalingDecision:
    """Scaling decision with rationale."""
    resource_type: ResourceType
    action: str  # 'scale_up', 'scale_down', 'no_change'
    current_value: float
    target_value: float
    confidence: float
    rationale: str
    timestamp: datetime = field(default_factory=datetime.now)
    expected_benefit: Dict[str, float] = field(default_factory=dict)


@dataclass
class ResourceMetrics:
    """Resource utilization and performance metrics."""
    cpu_utilization: float = 0.0
    gpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    network_bandwidth: float = 0.0
    throughput_ops_per_sec: float = 0.0
    latency_ms: float = 0.0
    carbon_efficiency: float = 0.0  # ops per gram CO2
    cost_efficiency: float = 0.0    # ops per dollar
    timestamp: datetime = field(default_factory=datetime.now)


class AutoScalingOptimizer:
    """Advanced auto-scaling and optimization system."""
    
    def __init__(
        self,
        strategy: ScalingStrategy = ScalingStrategy.CARBON_AWARE,
        min_scaling_interval: int = 300,  # 5 minutes
        max_concurrent_changes: int = 2,
        scaling_sensitivity: float = 0.1,
        carbon_weight: float = 0.6,
        performance_weight: float = 0.3,
        cost_weight: float = 0.1
    ):
        """Initialize auto-scaling optimizer.
        
        Args:
            strategy: Primary scaling strategy
            min_scaling_interval: Minimum time between scaling decisions (seconds)
            max_concurrent_changes: Maximum concurrent resource changes
            scaling_sensitivity: Sensitivity to trigger scaling (0.0-1.0)
            carbon_weight: Weight for carbon optimization in decisions
            performance_weight: Weight for performance optimization
            cost_weight: Weight for cost optimization
        """
        self.strategy = strategy
        self.min_scaling_interval = min_scaling_interval
        self.max_concurrent_changes = max_concurrent_changes
        self.scaling_sensitivity = scaling_sensitivity
        self.carbon_weight = carbon_weight
        self.performance_weight = performance_weight
        self.cost_weight = cost_weight
        
        # Scaling state
        self._active_changes: Dict[ResourceType, datetime] = {}
        self._resource_history: Dict[ResourceType, List[ResourceMetrics]] = {}
        self._scaling_history: List[ScalingDecision] = []
        self._performance_baseline: Optional[ResourceMetrics] = None
        
        # Optimization parameters
        self._resource_configs = {
            ResourceType.BATCH_SIZE: {
                'min_value': 1,
                'max_value': 512, 
                'step_size': 8,
                'default': 32
            },
            ResourceType.LEARNING_RATE: {
                'min_value': 1e-6,
                'max_value': 1e-1,
                'step_size': 'exponential',
                'default': 1e-3
            },
            ResourceType.PARALLELISM: {
                'min_value': 1,
                'max_value': 16,
                'step_size': 1,
                'default': 1
            },
            ResourceType.GPU_DEVICES: {
                'min_value': 1,
                'max_value': 8,
                'step_size': 1, 
                'default': 1
            }
        }
        
        # Thread pool for parallel optimization
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._optimization_lock = threading.RLock()
        
        # Performance tracking
        self._optimization_metrics = {
            'total_decisions': 0,
            'successful_optimizations': 0,
            'carbon_savings_kg': 0.0,
            'performance_improvements': 0.0,
            'cost_savings_usd': 0.0
        }
    
    async def analyze_and_optimize(
        self,
        current_metrics: ResourceMetrics,
        carbon_intensity: CarbonIntensity,
        training_state: TrainingState
    ) -> List[ScalingDecision]:
        """Analyze current state and generate scaling recommendations.
        
        Args:
            current_metrics: Current resource utilization metrics
            carbon_intensity: Current carbon intensity data
            training_state: Current training state
            
        Returns:
            List of scaling decisions to implement
        """
        decisions = []
        
        # Record current metrics
        await self._record_metrics(current_metrics)
        
        # Skip optimization if training is not active
        if training_state not in [TrainingState.RUNNING, TrainingState.PAUSED]:
            return decisions
        
        # Check if we can make scaling decisions
        if not self._can_make_scaling_decisions():
            return decisions
        
        try:
            with self._optimization_lock:
                # Generate scaling decisions based on strategy
                if self.strategy == ScalingStrategy.CARBON_AWARE:
                    decisions = await self._carbon_aware_optimization(
                        current_metrics, carbon_intensity
                    )
                elif self.strategy == ScalingStrategy.PERFORMANCE_BASED:
                    decisions = await self._performance_based_optimization(
                        current_metrics
                    )
                elif self.strategy == ScalingStrategy.COST_OPTIMIZED:
                    decisions = await self._cost_optimized_scaling(
                        current_metrics, carbon_intensity
                    )
                elif self.strategy == ScalingStrategy.HYBRID:
                    decisions = await self._hybrid_optimization(
                        current_metrics, carbon_intensity
                    )
                
                # Filter and validate decisions
                decisions = self._filter_scaling_decisions(decisions)
                
                # Update tracking
                for decision in decisions:
                    self._active_changes[decision.resource_type] = datetime.now()
                    self._scaling_history.append(decision)
                
                self._optimization_metrics['total_decisions'] += len(decisions)
        
        except Exception as e:
            logger.error(f"Optimization analysis failed: {e}")
        
        return decisions
    
    async def _carbon_aware_optimization(
        self,
        metrics: ResourceMetrics,
        carbon_intensity: CarbonIntensity
    ) -> List[ScalingDecision]:
        """Carbon-aware scaling optimization."""
        decisions = []
        
        # Get current carbon efficiency
        carbon_g_per_kwh = carbon_intensity.carbon_intensity
        current_efficiency = metrics.carbon_efficiency
        
        # Target different optimization based on carbon intensity
        if carbon_g_per_kwh > 200:  # High carbon intensity
            # Optimize for minimal resource usage
            decisions.extend(await self._minimize_resource_usage(metrics))
        elif carbon_g_per_kwh < 100:  # Low carbon intensity  
            # Optimize for maximum throughput
            decisions.extend(await self._maximize_throughput(metrics))
        else:
            # Balanced optimization
            decisions.extend(await self._balanced_optimization(metrics))
        
        # Add carbon-specific rationale
        for decision in decisions:
            decision.rationale += f" (Carbon: {carbon_g_per_kwh:.0f} gCO2/kWh)"
            decision.expected_benefit['carbon_savings_g'] = self._estimate_carbon_savings(
                decision, carbon_g_per_kwh
            )
        
        return decisions
    
    async def _performance_based_optimization(
        self,
        metrics: ResourceMetrics
    ) -> List[ScalingDecision]:
        """Performance-focused scaling optimization."""
        decisions = []
        
        # Analyze performance bottlenecks
        if metrics.gpu_utilization < 0.7:
            # GPU underutilized - could increase batch size or complexity
            decisions.append(ScalingDecision(
                resource_type=ResourceType.BATCH_SIZE,
                action='scale_up',
                current_value=self._get_current_resource_value(ResourceType.BATCH_SIZE),
                target_value=self._calculate_optimal_batch_size(metrics),
                confidence=0.8,
                rationale="GPU underutilized, increasing batch size",
                expected_benefit={'throughput_improvement': 0.2}
            ))
        
        elif metrics.gpu_utilization > 0.95:
            # GPU overutilized - reduce load
            decisions.append(ScalingDecision(
                resource_type=ResourceType.BATCH_SIZE,
                action='scale_down',
                current_value=self._get_current_resource_value(ResourceType.BATCH_SIZE),
                target_value=self._calculate_optimal_batch_size(metrics),
                confidence=0.9,
                rationale="GPU overutilized, reducing batch size",
                expected_benefit={'stability_improvement': 0.3}
            ))
        
        # Memory optimization
        if metrics.memory_utilization > 0.9:
            decisions.append(ScalingDecision(
                resource_type=ResourceType.BATCH_SIZE,
                action='scale_down',
                current_value=self._get_current_resource_value(ResourceType.BATCH_SIZE),
                target_value=max(8, self._get_current_resource_value(ResourceType.BATCH_SIZE) * 0.8),
                confidence=0.95,
                rationale="Memory usage critical, reducing batch size",
                expected_benefit={'stability_improvement': 0.4}
            ))
        
        return decisions
    
    async def _cost_optimized_scaling(
        self,
        metrics: ResourceMetrics,
        carbon_intensity: CarbonIntensity
    ) -> List[ScalingDecision]:
        """Cost-optimized scaling decisions."""
        decisions = []
        
        # Use carbon intensity as proxy for energy costs
        energy_cost_factor = carbon_intensity.carbon_intensity / 100.0
        
        # If energy is expensive, optimize for efficiency
        if energy_cost_factor > 2.0:
            # Reduce resource usage during expensive periods
            if metrics.gpu_utilization < 0.9:
                decisions.append(ScalingDecision(
                    resource_type=ResourceType.BATCH_SIZE,
                    action='scale_down',
                    current_value=self._get_current_resource_value(ResourceType.BATCH_SIZE),
                    target_value=max(16, self._get_current_resource_value(ResourceType.BATCH_SIZE) * 0.8),
                    confidence=0.7,
                    rationale="High energy costs, optimizing for efficiency",
                    expected_benefit={'cost_savings_usd': energy_cost_factor * 0.1}
                ))
        
        elif energy_cost_factor < 1.0:
            # Energy is cheap, optimize for speed
            decisions.append(ScalingDecision(
                resource_type=ResourceType.BATCH_SIZE,
                action='scale_up',
                current_value=self._get_current_resource_value(ResourceType.BATCH_SIZE),
                target_value=min(128, self._get_current_resource_value(ResourceType.BATCH_SIZE) * 1.2),
                confidence=0.6,
                rationale="Low energy costs, optimizing for speed",
                expected_benefit={'throughput_improvement': 0.15}
            ))
        
        return decisions
    
    async def _hybrid_optimization(
        self,
        metrics: ResourceMetrics,
        carbon_intensity: CarbonIntensity
    ) -> List[ScalingDecision]:
        """Hybrid optimization combining multiple strategies."""
        # Get decisions from all strategies
        carbon_decisions = await self._carbon_aware_optimization(metrics, carbon_intensity)
        perf_decisions = await self._performance_based_optimization(metrics)
        cost_decisions = await self._cost_optimized_scaling(metrics, carbon_intensity)
        
        # Combine and weight decisions
        all_decisions = carbon_decisions + perf_decisions + cost_decisions
        
        # Score decisions based on multi-objective optimization
        scored_decisions = []
        for decision in all_decisions:
            score = self._calculate_decision_score(decision, metrics, carbon_intensity)
            if score > 0.5:  # Only keep high-scoring decisions
                decision.confidence = score
                scored_decisions.append(decision)
        
        # Remove duplicates and conflicts
        final_decisions = self._resolve_decision_conflicts(scored_decisions)
        
        return final_decisions
    
    async def _minimize_resource_usage(self, metrics: ResourceMetrics) -> List[ScalingDecision]:
        """Generate decisions to minimize resource usage."""
        decisions = []
        
        # Reduce batch size if possible
        current_batch = self._get_current_resource_value(ResourceType.BATCH_SIZE)
        if current_batch > 16:
            decisions.append(ScalingDecision(
                resource_type=ResourceType.BATCH_SIZE,
                action='scale_down',
                current_value=current_batch,
                target_value=max(16, current_batch * 0.8),
                confidence=0.7,
                rationale="Minimizing resource usage for low carbon"
            ))
        
        return decisions
    
    async def _maximize_throughput(self, metrics: ResourceMetrics) -> List[ScalingDecision]:
        """Generate decisions to maximize throughput."""
        decisions = []
        
        # Increase batch size if GPU can handle it
        if metrics.gpu_utilization < 0.8 and metrics.memory_utilization < 0.8:
            current_batch = self._get_current_resource_value(ResourceType.BATCH_SIZE)
            decisions.append(ScalingDecision(
                resource_type=ResourceType.BATCH_SIZE,
                action='scale_up',
                current_value=current_batch,
                target_value=min(128, current_batch * 1.25),
                confidence=0.8,
                rationale="Maximizing throughput during low carbon periods"
            ))
        
        return decisions
    
    async def _balanced_optimization(self, metrics: ResourceMetrics) -> List[ScalingDecision]:
        """Generate balanced optimization decisions."""
        decisions = []
        
        # Aim for ~80% utilization
        target_utilization = 0.8
        
        if metrics.gpu_utilization < target_utilization - 0.1:
            current_batch = self._get_current_resource_value(ResourceType.BATCH_SIZE)
            decisions.append(ScalingDecision(
                resource_type=ResourceType.BATCH_SIZE,
                action='scale_up',
                current_value=current_batch,
                target_value=min(96, current_batch * 1.1),
                confidence=0.6,
                rationale="Balanced optimization - increasing utilization"
            ))
        elif metrics.gpu_utilization > target_utilization + 0.1:
            current_batch = self._get_current_resource_value(ResourceType.BATCH_SIZE)
            decisions.append(ScalingDecision(
                resource_type=ResourceType.BATCH_SIZE,
                action='scale_down', 
                current_value=current_batch,
                target_value=max(16, current_batch * 0.9),
                confidence=0.6,
                rationale="Balanced optimization - reducing utilization"
            ))
        
        return decisions
    
    def _calculate_decision_score(
        self,
        decision: ScalingDecision,
        metrics: ResourceMetrics,
        carbon_intensity: CarbonIntensity
    ) -> float:
        """Calculate multi-objective score for a scaling decision."""
        score = 0.0
        
        # Carbon efficiency score
        carbon_score = 0.0
        if 'carbon_savings_g' in decision.expected_benefit:
            carbon_score = min(1.0, decision.expected_benefit['carbon_savings_g'] / 100.0)
        
        # Performance score
        perf_score = 0.0
        if 'throughput_improvement' in decision.expected_benefit:
            perf_score = min(1.0, decision.expected_benefit['throughput_improvement'])
        
        # Cost score
        cost_score = 0.0
        if 'cost_savings_usd' in decision.expected_benefit:
            cost_score = min(1.0, decision.expected_benefit['cost_savings_usd'] / 10.0)
        
        # Weighted combination
        score = (
            self.carbon_weight * carbon_score +
            self.performance_weight * perf_score +
            self.cost_weight * cost_score
        )
        
        # Apply confidence factor
        score *= decision.confidence
        
        return score
    
    def _resolve_decision_conflicts(
        self,
        decisions: List[ScalingDecision]
    ) -> List[ScalingDecision]:
        """Resolve conflicting scaling decisions."""
        # Group by resource type
        by_resource = {}
        for decision in decisions:
            resource = decision.resource_type
            if resource not in by_resource:
                by_resource[resource] = []
            by_resource[resource].append(decision)
        
        # For each resource, keep only the highest scored decision
        final_decisions = []
        for resource, resource_decisions in by_resource.items():
            if len(resource_decisions) == 1:
                final_decisions.append(resource_decisions[0])
            else:
                # Pick the highest confidence decision
                best_decision = max(resource_decisions, key=lambda d: d.confidence)
                final_decisions.append(best_decision)
        
        return final_decisions
    
    def _filter_scaling_decisions(
        self,
        decisions: List[ScalingDecision]
    ) -> List[ScalingDecision]:
        """Filter scaling decisions based on constraints."""
        filtered = []
        
        for decision in decisions:
            # Check if resource is already being changed
            if decision.resource_type in self._active_changes:
                last_change = self._active_changes[decision.resource_type]
                if (datetime.now() - last_change).total_seconds() < self.min_scaling_interval:
                    continue
            
            # Check if we've exceeded max concurrent changes
            if len(self._active_changes) >= self.max_concurrent_changes:
                break
            
            # Validate resource bounds
            if not self._validate_resource_bounds(decision):
                continue
            
            filtered.append(decision)
        
        return filtered
    
    def _validate_resource_bounds(self, decision: ScalingDecision) -> bool:
        """Validate that scaling decision is within resource bounds."""
        config = self._resource_configs.get(decision.resource_type)
        if not config:
            return True
        
        target = decision.target_value
        return config['min_value'] <= target <= config['max_value']
    
    def _can_make_scaling_decisions(self) -> bool:
        """Check if system can make scaling decisions."""
        # Don't overwhelm the system with too many concurrent changes
        active_count = len([
            change_time for change_time in self._active_changes.values()
            if (datetime.now() - change_time).total_seconds() < self.min_scaling_interval * 2
        ])
        
        return active_count < self.max_concurrent_changes
    
    def _get_current_resource_value(self, resource_type: ResourceType) -> float:
        """Get current value for a resource type."""
        # This would integrate with actual resource management system
        # For now, return defaults
        return self._resource_configs.get(resource_type, {}).get('default', 1.0)
    
    def _calculate_optimal_batch_size(self, metrics: ResourceMetrics) -> float:
        """Calculate optimal batch size based on current metrics."""
        current_batch = self._get_current_resource_value(ResourceType.BATCH_SIZE)
        
        # Simple heuristic based on GPU utilization
        if metrics.gpu_utilization < 0.5:
            return min(128, current_batch * 1.5)
        elif metrics.gpu_utilization > 0.9:
            return max(8, current_batch * 0.7)
        else:
            return current_batch
    
    def _estimate_carbon_savings(
        self,
        decision: ScalingDecision,
        carbon_intensity: float
    ) -> float:
        """Estimate carbon savings from a scaling decision."""
        # Simple estimation based on resource change
        resource_change_pct = (decision.target_value - decision.current_value) / decision.current_value
        
        # Assume roughly linear relationship for small changes
        estimated_energy_change = resource_change_pct * 0.5  # 50% efficiency
        estimated_carbon_savings = abs(estimated_energy_change) * carbon_intensity * 0.1  # 0.1 kWh baseline
        
        return estimated_carbon_savings
    
    async def _record_metrics(self, metrics: ResourceMetrics) -> None:
        """Record resource metrics for historical analysis."""
        # Keep history for analysis
        for resource_type in ResourceType:
            if resource_type not in self._resource_history:
                self._resource_history[resource_type] = []
            
            # Keep last 100 measurements
            history = self._resource_history[resource_type]
            history.append(metrics)
            if len(history) > 100:
                history.pop(0)
        
        # Update baseline if needed
        if not self._performance_baseline:
            self._performance_baseline = metrics
    
    def complete_scaling_change(
        self,
        resource_type: ResourceType,
        success: bool,
        actual_benefit: Optional[Dict[str, float]] = None
    ) -> None:
        """Mark a scaling change as completed."""
        if resource_type in self._active_changes:
            del self._active_changes[resource_type]
        
        if success:
            self._optimization_metrics['successful_optimizations'] += 1
            
            # Record benefits if provided
            if actual_benefit:
                if 'carbon_savings_kg' in actual_benefit:
                    self._optimization_metrics['carbon_savings_kg'] += actual_benefit['carbon_savings_kg']
                if 'performance_improvement' in actual_benefit:
                    self._optimization_metrics['performance_improvements'] += actual_benefit['performance_improvement']
                if 'cost_savings_usd' in actual_benefit:
                    self._optimization_metrics['cost_savings_usd'] += actual_benefit['cost_savings_usd']
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization performance."""
        success_rate = 0.0
        if self._optimization_metrics['total_decisions'] > 0:
            success_rate = (
                self._optimization_metrics['successful_optimizations'] / 
                self._optimization_metrics['total_decisions']
            )
        
        return {
            'strategy': self.strategy.value,
            'total_decisions': self._optimization_metrics['total_decisions'],
            'successful_optimizations': self._optimization_metrics['successful_optimizations'],
            'success_rate': success_rate,
            'active_changes': len(self._active_changes),
            'carbon_savings_kg': self._optimization_metrics['carbon_savings_kg'],
            'performance_improvements': self._optimization_metrics['performance_improvements'],
            'cost_savings_usd': self._optimization_metrics['cost_savings_usd'],
            'recent_decisions': len([
                d for d in self._scaling_history 
                if d.timestamp > datetime.now() - timedelta(hours=1)
            ])
        }