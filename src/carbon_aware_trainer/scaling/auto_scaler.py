"""Auto-scaling system for carbon-aware training."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging

from ..core.types import CarbonIntensity, TrainingMetrics
from ..core.monitor import CarbonMonitor

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Scaling direction."""
    UP = "up"
    DOWN = "down"
    NONE = "none"


class ScalingTrigger(Enum):
    """Scaling trigger types."""
    CARBON_BASED = "carbon_based"
    COST_BASED = "cost_based"
    PERFORMANCE_BASED = "performance_based"
    RENEWABLE_BASED = "renewable_based"
    HYBRID = "hybrid"


@dataclass
class ScalingPolicy:
    """Configuration for auto-scaling policies."""
    
    # Scaling thresholds
    scale_up_carbon_threshold: float = 80.0  # gCO2/kWh
    scale_down_carbon_threshold: float = 150.0  # gCO2/kWh
    
    # Cost-based scaling
    max_cost_per_hour_usd: float = 50.0
    cost_optimization_weight: float = 0.3
    
    # Performance thresholds
    min_renewable_percentage: float = 30.0
    max_training_delay_hours: float = 24.0
    
    # Scaling parameters
    min_instances: int = 1
    max_instances: int = 10
    scale_up_factor: float = 1.5  # Multiply instances by this factor
    scale_down_factor: float = 0.7
    
    # Cooldown periods
    scale_up_cooldown_minutes: int = 10
    scale_down_cooldown_minutes: int = 15
    
    # Decision making
    trigger_type: ScalingTrigger = ScalingTrigger.HYBRID
    decision_window_minutes: int = 5  # Window for metric aggregation


@dataclass 
class ScalingMetrics:
    """Current scaling metrics."""
    
    timestamp: datetime
    current_instances: int
    target_instances: int
    carbon_intensity: float
    renewable_percentage: float
    cost_per_hour: float
    training_efficiency: float
    
    # Scaling decision
    scaling_decision: ScalingDirection
    trigger_reason: str
    confidence_score: float
    
    # Resource utilization
    cpu_utilization: float = 0.0
    gpu_utilization: float = 0.0
    memory_utilization: float = 0.0


@dataclass
class ScalingAction:
    """Scaling action to be executed."""
    
    action_id: str
    timestamp: datetime
    direction: ScalingDirection
    from_instances: int
    to_instances: int
    trigger_reason: str
    estimated_cost_impact: float
    estimated_carbon_impact: float
    region: str


class AutoScaler:
    """Advanced auto-scaling system with carbon awareness."""
    
    def __init__(
        self,
        policy: ScalingPolicy,
        carbon_monitor: CarbonMonitor,
        instance_manager: Optional[Callable] = None
    ):
        """Initialize auto-scaler.
        
        Args:
            policy: Scaling policy configuration
            carbon_monitor: Carbon monitoring instance
            instance_manager: Callback for managing compute instances
        """
        self.policy = policy
        self.carbon_monitor = carbon_monitor
        self.instance_manager = instance_manager
        
        # State tracking
        self.current_instances = policy.min_instances
        self.target_instances = policy.min_instances
        self.metrics_history: List[ScalingMetrics] = []
        self.scaling_actions: List[ScalingAction] = []
        
        # Cooldown tracking
        self.last_scale_up_time: Optional[datetime] = None
        self.last_scale_down_time: Optional[datetime] = None
        
        # Auto-scaling control
        self._scaling_task: Optional[asyncio.Task] = None
        self._is_running = False
        
        # Performance predictors
        self.carbon_predictor = CarbonScalingPredictor()
        self.cost_optimizer = CostOptimizer()
    
    async def start_auto_scaling(self, check_interval_minutes: int = 2) -> None:
        """Start the auto-scaling system.
        
        Args:
            check_interval_minutes: How often to evaluate scaling decisions
        """
        if self._is_running:
            logger.warning("Auto-scaling already running")
            return
        
        self._is_running = True
        self._scaling_task = asyncio.create_task(
            self._scaling_loop(check_interval_minutes * 60)
        )
        logger.info(f"Auto-scaling started (check every {check_interval_minutes}m)")
    
    async def stop_auto_scaling(self) -> None:
        """Stop the auto-scaling system."""
        self._is_running = False
        
        if self._scaling_task and not self._scaling_task.done():
            self._scaling_task.cancel()
            try:
                await self._scaling_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Auto-scaling stopped")
    
    async def _scaling_loop(self, check_interval_seconds: int) -> None:
        """Main auto-scaling decision loop."""
        while self._is_running:
            try:
                # Collect current metrics
                current_metrics = await self._collect_scaling_metrics()
                
                if current_metrics:
                    # Make scaling decision
                    scaling_decision = await self._make_scaling_decision(current_metrics)
                    current_metrics.scaling_decision = scaling_decision.direction
                    current_metrics.trigger_reason = scaling_decision.trigger_reason
                    current_metrics.confidence_score = scaling_decision.confidence_score
                    
                    # Store metrics
                    self.metrics_history.append(current_metrics)
                    self._cleanup_old_metrics()
                    
                    # Execute scaling action if needed
                    if scaling_decision.direction != ScalingDirection.NONE:
                        await self._execute_scaling_action(scaling_decision)
                
                await asyncio.sleep(check_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in auto-scaling loop: {e}")
                await asyncio.sleep(30)  # Wait before retrying
    
    async def _collect_scaling_metrics(self) -> Optional[ScalingMetrics]:
        """Collect current metrics for scaling decisions."""
        try:
            # Get carbon data
            primary_region = self.carbon_monitor.regions[0] if self.carbon_monitor.regions else "US-CA"
            carbon_intensity = await self.carbon_monitor.get_current_intensity(primary_region)
            
            if not carbon_intensity:
                return None
            
            # Calculate current cost (simplified)
            current_cost = self._calculate_current_cost_per_hour(primary_region)
            
            # Estimate training efficiency
            training_efficiency = self._estimate_training_efficiency()
            
            # Get resource utilization (would integrate with actual monitoring)
            resource_utilization = self._get_resource_utilization()
            
            metrics = ScalingMetrics(
                timestamp=datetime.now(),
                current_instances=self.current_instances,
                target_instances=self.target_instances,
                carbon_intensity=carbon_intensity.carbon_intensity,
                renewable_percentage=carbon_intensity.renewable_percentage or 0,
                cost_per_hour=current_cost,
                training_efficiency=training_efficiency,
                scaling_decision=ScalingDirection.NONE,
                trigger_reason="",
                confidence_score=0.0,
                cpu_utilization=resource_utilization.get('cpu', 0),
                gpu_utilization=resource_utilization.get('gpu', 0),
                memory_utilization=resource_utilization.get('memory', 0)
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting scaling metrics: {e}")
            return None
    
    async def _make_scaling_decision(self, metrics: ScalingMetrics) -> 'ScalingDecision':
        """Make intelligent scaling decision based on multiple factors.
        
        Args:
            metrics: Current scaling metrics
            
        Returns:
            Scaling decision with reasoning
        """
        decision_factors = []
        
        # Carbon-based scaling
        carbon_decision = self._evaluate_carbon_scaling(metrics)
        decision_factors.append(carbon_decision)
        
        # Cost-based scaling
        cost_decision = self._evaluate_cost_scaling(metrics)
        decision_factors.append(cost_decision)
        
        # Performance-based scaling
        performance_decision = self._evaluate_performance_scaling(metrics)
        decision_factors.append(performance_decision)
        
        # Renewable energy scaling
        renewable_decision = self._evaluate_renewable_scaling(metrics)
        decision_factors.append(renewable_decision)
        
        # Combine decisions using policy
        final_decision = self._combine_scaling_decisions(decision_factors)
        
        # Apply cooldown and safety checks
        final_decision = self._apply_scaling_constraints(final_decision, metrics)
        
        return final_decision
    
    def _evaluate_carbon_scaling(self, metrics: ScalingMetrics) -> 'ScalingDecision':
        """Evaluate scaling based on carbon intensity."""
        carbon_intensity = metrics.carbon_intensity
        confidence = 0.8
        
        if carbon_intensity <= self.policy.scale_up_carbon_threshold:
            # Low carbon - consider scaling up
            return ScalingDecision(
                direction=ScalingDirection.UP,
                trigger_reason=f"Low carbon intensity ({carbon_intensity:.1f} gCO2/kWh)",
                confidence_score=confidence,
                weight=1.0
            )
        elif carbon_intensity >= self.policy.scale_down_carbon_threshold:
            # High carbon - consider scaling down
            return ScalingDecision(
                direction=ScalingDirection.DOWN,
                trigger_reason=f"High carbon intensity ({carbon_intensity:.1f} gCO2/kWh)",
                confidence_score=confidence,
                weight=1.0
            )
        else:
            return ScalingDecision(
                direction=ScalingDirection.NONE,
                trigger_reason="Carbon intensity within acceptable range",
                confidence_score=0.5,
                weight=0.0
            )
    
    def _evaluate_cost_scaling(self, metrics: ScalingMetrics) -> 'ScalingDecision':
        """Evaluate scaling based on cost optimization."""
        current_cost = metrics.cost_per_hour
        max_cost = self.policy.max_cost_per_hour_usd
        
        if current_cost > max_cost:
            return ScalingDecision(
                direction=ScalingDirection.DOWN,
                trigger_reason=f"Cost too high (${current_cost:.2f}/hr > ${max_cost:.2f}/hr)",
                confidence_score=0.9,
                weight=self.policy.cost_optimization_weight
            )
        elif current_cost < max_cost * 0.5:  # Very low cost - consider scaling up
            return ScalingDecision(
                direction=ScalingDirection.UP,
                trigger_reason=f"Cost efficient (${current_cost:.2f}/hr < ${max_cost*0.5:.2f}/hr)",
                confidence_score=0.6,
                weight=self.policy.cost_optimization_weight * 0.5
            )
        else:
            return ScalingDecision(
                direction=ScalingDirection.NONE,
                trigger_reason="Cost within acceptable range",
                confidence_score=0.5,
                weight=0.0
            )
    
    def _evaluate_performance_scaling(self, metrics: ScalingMetrics) -> 'ScalingDecision':
        """Evaluate scaling based on performance metrics."""
        gpu_util = metrics.gpu_utilization
        
        if gpu_util > 90:
            # High utilization - scale up
            return ScalingDecision(
                direction=ScalingDirection.UP,
                trigger_reason=f"High GPU utilization ({gpu_util:.1f}%)",
                confidence_score=0.8,
                weight=0.7
            )
        elif gpu_util < 30 and self.current_instances > self.policy.min_instances:
            # Low utilization - scale down
            return ScalingDecision(
                direction=ScalingDirection.DOWN,
                trigger_reason=f"Low GPU utilization ({gpu_util:.1f}%)",
                confidence_score=0.7,
                weight=0.5
            )
        else:
            return ScalingDecision(
                direction=ScalingDirection.NONE,
                trigger_reason="Performance metrics within acceptable range",
                confidence_score=0.5,
                weight=0.0
            )
    
    def _evaluate_renewable_scaling(self, metrics: ScalingMetrics) -> 'ScalingDecision':
        """Evaluate scaling based on renewable energy availability."""
        renewable_pct = metrics.renewable_percentage
        min_renewable = self.policy.min_renewable_percentage
        
        if renewable_pct >= min_renewable * 1.5:
            # High renewable - consider scaling up
            return ScalingDecision(
                direction=ScalingDirection.UP,
                trigger_reason=f"High renewable energy ({renewable_pct:.1f}%)",
                confidence_score=0.7,
                weight=0.6
            )
        elif renewable_pct < min_renewable:
            # Low renewable - consider scaling down
            return ScalingDecision(
                direction=ScalingDirection.DOWN,
                trigger_reason=f"Low renewable energy ({renewable_pct:.1f}%)",
                confidence_score=0.6,
                weight=0.4
            )
        else:
            return ScalingDecision(
                direction=ScalingDirection.NONE,
                trigger_reason="Renewable energy within acceptable range",
                confidence_score=0.5,
                weight=0.0
            )
    
    def _combine_scaling_decisions(self, decisions: List['ScalingDecision']) -> 'ScalingDecision':
        """Combine multiple scaling decisions into final decision."""
        # Calculate weighted scores
        scale_up_score = sum(
            d.weight * d.confidence_score 
            for d in decisions 
            if d.direction == ScalingDirection.UP
        )
        
        scale_down_score = sum(
            d.weight * d.confidence_score 
            for d in decisions 
            if d.direction == ScalingDirection.DOWN
        )
        
        # Collect reasons
        up_reasons = [d.trigger_reason for d in decisions if d.direction == ScalingDirection.UP]
        down_reasons = [d.trigger_reason for d in decisions if d.direction == ScalingDirection.DOWN]
        
        # Make final decision
        if scale_up_score > scale_down_score and scale_up_score > 0.5:
            return ScalingDecision(
                direction=ScalingDirection.UP,
                trigger_reason=f"Scale up: {'; '.join(up_reasons)}",
                confidence_score=min(scale_up_score, 1.0),
                weight=1.0
            )
        elif scale_down_score > scale_up_score and scale_down_score > 0.5:
            return ScalingDecision(
                direction=ScalingDirection.DOWN,
                trigger_reason=f"Scale down: {'; '.join(down_reasons)}",
                confidence_score=min(scale_down_score, 1.0),
                weight=1.0
            )
        else:
            return ScalingDecision(
                direction=ScalingDirection.NONE,
                trigger_reason="No strong scaling signal detected",
                confidence_score=0.5,
                weight=0.0
            )
    
    def _apply_scaling_constraints(
        self,
        decision: 'ScalingDecision',
        metrics: ScalingMetrics
    ) -> 'ScalingDecision':
        """Apply cooldown periods and safety constraints."""
        now = datetime.now()
        
        # Check cooldown periods
        if decision.direction == ScalingDirection.UP:
            if (self.last_scale_up_time and 
                now - self.last_scale_up_time < timedelta(minutes=self.policy.scale_up_cooldown_minutes)):
                return ScalingDecision(
                    direction=ScalingDirection.NONE,
                    trigger_reason="Scale up cooldown period active",
                    confidence_score=0.0,
                    weight=0.0
                )
        
        elif decision.direction == ScalingDirection.DOWN:
            if (self.last_scale_down_time and 
                now - self.last_scale_down_time < timedelta(minutes=self.policy.scale_down_cooldown_minutes)):
                return ScalingDecision(
                    direction=ScalingDirection.NONE,
                    trigger_reason="Scale down cooldown period active",
                    confidence_score=0.0,
                    weight=0.0
                )
        
        # Check instance limits
        if decision.direction == ScalingDirection.UP:
            if self.current_instances >= self.policy.max_instances:
                return ScalingDecision(
                    direction=ScalingDirection.NONE,
                    trigger_reason=f"Already at maximum instances ({self.policy.max_instances})",
                    confidence_score=0.0,
                    weight=0.0
                )
        
        elif decision.direction == ScalingDirection.DOWN:
            if self.current_instances <= self.policy.min_instances:
                return ScalingDecision(
                    direction=ScalingDirection.NONE,
                    trigger_reason=f"Already at minimum instances ({self.policy.min_instances})",
                    confidence_score=0.0,
                    weight=0.0
                )
        
        return decision
    
    async def _execute_scaling_action(self, decision: 'ScalingDecision') -> bool:
        """Execute the scaling action.
        
        Args:
            decision: Scaling decision to execute
            
        Returns:
            True if scaling action was successful
        """
        try:
            # Calculate new instance count
            if decision.direction == ScalingDirection.UP:
                new_instances = min(
                    int(self.current_instances * self.policy.scale_up_factor),
                    self.policy.max_instances
                )
                self.last_scale_up_time = datetime.now()
                
            elif decision.direction == ScalingDirection.DOWN:
                new_instances = max(
                    int(self.current_instances * self.policy.scale_down_factor),
                    self.policy.min_instances
                )
                self.last_scale_down_time = datetime.now()
            else:
                return True  # No action needed
            
            if new_instances == self.current_instances:
                return True  # No change needed
            
            # Create scaling action record
            action = ScalingAction(
                action_id=f"scale_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),
                direction=decision.direction,
                from_instances=self.current_instances,
                to_instances=new_instances,
                trigger_reason=decision.trigger_reason,
                estimated_cost_impact=self._estimate_cost_impact(self.current_instances, new_instances),
                estimated_carbon_impact=self._estimate_carbon_impact(self.current_instances, new_instances),
                region=self.carbon_monitor.regions[0] if self.carbon_monitor.regions else "US-CA"
            )
            
            logger.info(
                f"Executing scaling action: {self.current_instances} -> {new_instances} instances"
                f" (Reason: {decision.trigger_reason})"
            )
            
            # Execute actual scaling (via instance manager callback)
            if self.instance_manager:
                success = await self.instance_manager(
                    action_type="scale",
                    from_instances=self.current_instances,
                    to_instances=new_instances,
                    action_data=action
                )
                
                if success:
                    self.current_instances = new_instances
                    self.target_instances = new_instances
                    self.scaling_actions.append(action)
                    
                    logger.info(f"Scaling successful: now running {new_instances} instances")
                    return True
                else:
                    logger.error(f"Scaling failed: instance manager returned failure")
                    return False
            else:
                # No instance manager - just update internal state
                self.current_instances = new_instances
                self.target_instances = new_instances
                self.scaling_actions.append(action)
                
                logger.info(f"Scaling completed (simulated): now {new_instances} instances")
                return True
                
        except Exception as e:
            logger.error(f"Error executing scaling action: {e}")
            return False
    
    def _calculate_current_cost_per_hour(self, region: str) -> float:
        """Calculate current cost per hour for the region."""
        # Simplified cost calculation - would integrate with actual cloud pricing
        base_cost_per_instance = {
            'US-CA': 2.50,
            'US-TX': 2.20,
            'EU-FR': 2.80,
            'EU-DE': 3.20,
            'CN-GD': 1.80
        }.get(region, 2.50)
        
        return base_cost_per_instance * self.current_instances
    
    def _estimate_training_efficiency(self) -> float:
        """Estimate current training efficiency."""
        # Simplified efficiency calculation
        if self.metrics_history:
            recent_metrics = self.metrics_history[-5:]  # Last 5 data points
            avg_gpu_util = sum(m.gpu_utilization for m in recent_metrics) / len(recent_metrics)
            return avg_gpu_util / 100.0
        return 0.8  # Default efficiency
    
    def _get_resource_utilization(self) -> Dict[str, float]:
        """Get current resource utilization metrics."""
        # Simplified - would integrate with actual monitoring systems
        import random
        return {
            'cpu': random.uniform(40, 90),
            'gpu': random.uniform(50, 95),
            'memory': random.uniform(30, 80)
        }
    
    def _estimate_cost_impact(self, from_instances: int, to_instances: int) -> float:
        """Estimate cost impact of scaling action."""
        instance_cost_per_hour = 2.50  # Simplified
        cost_change_per_hour = (to_instances - from_instances) * instance_cost_per_hour
        return cost_change_per_hour
    
    def _estimate_carbon_impact(self, from_instances: int, to_instances: int) -> float:
        """Estimate carbon impact of scaling action."""
        power_per_instance_kw = 0.4  # 400W GPU
        carbon_intensity = (
            self.metrics_history[-1].carbon_intensity 
            if self.metrics_history else 200
        )
        
        power_change_kw = (to_instances - from_instances) * power_per_instance_kw
        carbon_impact_kg_per_hour = power_change_kw * (carbon_intensity / 1000)
        return carbon_impact_kg_per_hour
    
    def _cleanup_old_metrics(self) -> None:
        """Remove old metrics beyond retention period."""
        cutoff_time = datetime.now() - timedelta(hours=24)  # Keep 24 hours
        self.metrics_history = [
            m for m in self.metrics_history
            if m.timestamp >= cutoff_time
        ]
        
        # Keep scaling actions for longer (7 days)
        cutoff_time = datetime.now() - timedelta(days=7)
        self.scaling_actions = [
            a for a in self.scaling_actions
            if a.timestamp >= cutoff_time
        ]
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current auto-scaling status.
        
        Returns:
            Dictionary with scaling status information
        """
        current_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        return {
            'is_running': self._is_running,
            'current_instances': self.current_instances,
            'target_instances': self.target_instances,
            'policy': {
                'min_instances': self.policy.min_instances,
                'max_instances': self.policy.max_instances,
                'carbon_thresholds': {
                    'scale_up': self.policy.scale_up_carbon_threshold,
                    'scale_down': self.policy.scale_down_carbon_threshold
                }
            },
            'current_metrics': {
                'carbon_intensity': current_metrics.carbon_intensity if current_metrics else 0,
                'cost_per_hour': current_metrics.cost_per_hour if current_metrics else 0,
                'gpu_utilization': current_metrics.gpu_utilization if current_metrics else 0
            } if current_metrics else None,
            'recent_actions': len([
                a for a in self.scaling_actions
                if a.timestamp >= datetime.now() - timedelta(hours=1)
            ]),
            'total_actions': len(self.scaling_actions),
            'last_scale_up': self.last_scale_up_time.isoformat() if self.last_scale_up_time else None,
            'last_scale_down': self.last_scale_down_time.isoformat() if self.last_scale_down_time else None
        }


@dataclass
class ScalingDecision:
    """Internal scaling decision representation."""
    direction: ScalingDirection
    trigger_reason: str
    confidence_score: float
    weight: float


class CarbonScalingPredictor:
    """Predictive model for carbon-aware scaling."""
    
    def predict_optimal_instances(
        self,
        carbon_forecast: List[CarbonIntensity],
        current_instances: int
    ) -> int:
        """Predict optimal instance count based on carbon forecast."""
        # Simplified prediction logic
        if not carbon_forecast:
            return current_instances
        
        avg_carbon = sum(ci.carbon_intensity for ci in carbon_forecast) / len(carbon_forecast)
        
        if avg_carbon < 80:  # Low carbon - can scale up
            return min(current_instances + 1, 10)
        elif avg_carbon > 200:  # High carbon - should scale down
            return max(current_instances - 1, 1)
        
        return current_instances


class CostOptimizer:
    """Cost optimization for auto-scaling."""
    
    def calculate_optimal_cost_instances(
        self,
        target_performance: float,
        cost_budget_per_hour: float,
        instance_cost: float
    ) -> int:
        """Calculate optimal instance count for cost constraints."""
        max_instances_by_cost = int(cost_budget_per_hour / instance_cost)
        min_instances_for_performance = max(1, int(target_performance * 2))
        
        return min(max_instances_by_cost, min_instances_for_performance)