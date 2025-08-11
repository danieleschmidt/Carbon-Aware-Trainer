"""
Intelligent auto-scaling system for carbon-aware training workloads.
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import math

import numpy as np

from .exceptions import CarbonAwareException

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Scaling direction."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class ScalingTrigger(Enum):
    """Triggers for scaling decisions."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    QUEUE_DEPTH = "queue_depth"
    RESPONSE_TIME = "response_time"
    CARBON_INTENSITY = "carbon_intensity"
    COST_OPTIMIZATION = "cost_optimization"
    DEMAND_FORECAST = "demand_forecast"


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""
    timestamp: datetime
    cpu_utilization: float
    memory_utilization: float
    gpu_utilization: float = 0.0
    queue_depth: int = 0
    active_requests: int = 0
    response_time_p95: float = 0.0
    error_rate: float = 0.0
    carbon_intensity: Optional[float] = None
    cost_per_hour: Optional[float] = None
    demand_forecast: Optional[float] = None


@dataclass
class ScalingRule:
    """Rule for scaling decisions."""
    name: str
    trigger: ScalingTrigger
    scale_up_threshold: float
    scale_down_threshold: float
    min_instances: int = 1
    max_instances: int = 10
    cooldown_minutes: int = 5
    weight: float = 1.0
    enabled: bool = True


@dataclass
class ScalingAction:
    """Scaling action to be executed."""
    direction: ScalingDirection
    target_instances: int
    current_instances: int
    triggered_by: List[str]
    confidence: float
    estimated_impact: Dict[str, float]
    rationale: str
    carbon_aware: bool = False


class PredictiveScaler:
    """Predictive scaling based on historical patterns and forecasts."""
    
    def __init__(self, lookback_hours: int = 24):
        self.lookback_hours = lookback_hours
        self.metrics_history: List[ScalingMetrics] = []
        self.scaling_history: List[Dict[str, Any]] = []
        
    def add_metrics(self, metrics: ScalingMetrics) -> None:
        """Add metrics to history."""
        self.metrics_history.append(metrics)
        
        # Keep history bounded
        cutoff = datetime.now() - timedelta(hours=self.lookback_hours)
        self.metrics_history = [
            m for m in self.metrics_history 
            if m.timestamp > cutoff
        ]
        
    def predict_demand(self, horizon_minutes: int = 30) -> Optional[float]:
        """Predict demand for the next period."""
        if len(self.metrics_history) < 10:
            return None
            
        # Extract time series data
        timestamps = [m.timestamp for m in self.metrics_history]
        cpu_values = [m.cpu_utilization for m in self.metrics_history]
        
        # Simple trend analysis (could be replaced with ML model)
        recent_values = cpu_values[-6:]  # Last 6 data points
        if len(recent_values) >= 3:
            # Calculate trend
            x = np.arange(len(recent_values))
            slope = np.polyfit(x, recent_values, 1)[0]
            
            # Project forward
            current_value = recent_values[-1]
            prediction = current_value + slope * (horizon_minutes / 5)  # Assuming 5-min intervals
            
            return max(0, min(100, prediction))  # Bound between 0-100%
            
        return None
        
    def detect_patterns(self) -> Dict[str, Any]:
        """Detect usage patterns from historical data."""
        if len(self.metrics_history) < 24:  # Need at least 24 data points
            return {"status": "insufficient_data"}
            
        # Analyze daily patterns
        hourly_cpu = {}
        hourly_memory = {}
        
        for metrics in self.metrics_history:
            hour = metrics.timestamp.hour
            if hour not in hourly_cpu:
                hourly_cpu[hour] = []
                hourly_memory[hour] = []
                
            hourly_cpu[hour].append(metrics.cpu_utilization)
            hourly_memory[hour].append(metrics.memory_utilization)
            
        # Calculate averages
        avg_cpu_by_hour = {
            hour: np.mean(values) for hour, values in hourly_cpu.items()
        }
        avg_memory_by_hour = {
            hour: np.mean(values) for hour, values in hourly_memory.items()
        }
        
        # Find peak and low usage hours
        if avg_cpu_by_hour:
            peak_hour = max(avg_cpu_by_hour.keys(), key=lambda h: avg_cpu_by_hour[h])
            low_hour = min(avg_cpu_by_hour.keys(), key=lambda h: avg_cpu_by_hour[h])
            
            return {
                "peak_hour": peak_hour,
                "peak_cpu": avg_cpu_by_hour[peak_hour],
                "low_hour": low_hour,
                "low_cpu": avg_cpu_by_hour[low_hour],
                "cpu_variance": np.var(list(avg_cpu_by_hour.values())),
                "memory_variance": np.var(list(avg_memory_by_hour.values())),
                "pattern_strength": "high" if np.var(list(avg_cpu_by_hour.values())) > 100 else "low"
            }
            
        return {"status": "no_patterns"}


class CarbonAwareScaler:
    """Scaling component that considers carbon intensity."""
    
    def __init__(self, carbon_weight: float = 0.3):
        self.carbon_weight = carbon_weight  # Weight of carbon in scaling decisions
        self.carbon_history: List[Tuple[datetime, float]] = []
        
    def should_scale_for_carbon(
        self,
        current_carbon: float,
        current_instances: int,
        max_instances: int
    ) -> Tuple[bool, ScalingDirection, str]:
        """Determine if scaling should occur based on carbon intensity."""
        
        # Record carbon data
        self.carbon_history.append((datetime.now(), current_carbon))
        
        # Keep last 24 hours of data
        cutoff = datetime.now() - timedelta(hours=24)
        self.carbon_history = [
            (ts, carbon) for ts, carbon in self.carbon_history
            if ts > cutoff
        ]
        
        # Low carbon intensity - scale up for efficiency
        if current_carbon < 50 and current_instances < max_instances:
            return True, ScalingDirection.UP, f"Low carbon intensity ({current_carbon:.1f} gCO2/kWh) - scale up for efficiency"
            
        # Very high carbon intensity - scale down to reduce emissions
        elif current_carbon > 300 and current_instances > 1:
            return True, ScalingDirection.DOWN, f"High carbon intensity ({current_carbon:.1f} gCO2/kWh) - scale down to reduce emissions"
            
        # Moderate carbon - check trend
        elif len(self.carbon_history) >= 3:
            recent_carbon = [carbon for _, carbon in self.carbon_history[-3:]]
            carbon_trend = np.polyfit(range(len(recent_carbon)), recent_carbon, 1)[0]
            
            # Carbon intensity rising rapidly - prepare to scale down
            if carbon_trend > 50 and current_carbon > 150:
                return True, ScalingDirection.DOWN, f"Carbon intensity rising rapidly (trend: +{carbon_trend:.1f}/hour)"
                
            # Carbon intensity dropping - can scale up
            elif carbon_trend < -30 and current_carbon < 200:
                return True, ScalingDirection.UP, f"Carbon intensity dropping (trend: {carbon_trend:.1f}/hour)"
                
        return False, ScalingDirection.STABLE, "Carbon conditions stable"
        
    def get_carbon_efficiency_score(self, instances: int) -> float:
        """Calculate carbon efficiency score for given instance count."""
        if not self.carbon_history:
            return 0.5  # Neutral score
            
        current_carbon = self.carbon_history[-1][1]
        
        # Lower carbon intensity = higher efficiency score
        # Scale efficiency by number of instances (more instances = more emissions)
        base_efficiency = max(0, (400 - current_carbon) / 400)  # Normalize 0-400 gCO2/kWh to 0-1
        instance_penalty = min(0.2, instances * 0.02)  # Small penalty for more instances
        
        return max(0, base_efficiency - instance_penalty)


class IntelligentAutoScaler:
    """Comprehensive auto-scaling system with carbon awareness."""
    
    def __init__(
        self,
        enable_predictive: bool = True,
        enable_carbon_aware: bool = True,
        carbon_weight: float = 0.3
    ):
        self.enable_predictive = enable_predictive
        self.enable_carbon_aware = enable_carbon_aware
        self.carbon_weight = carbon_weight
        
        # Components
        self.predictive_scaler = PredictiveScaler() if enable_predictive else None
        self.carbon_scaler = CarbonAwareScaler(carbon_weight) if enable_carbon_aware else None
        
        # Scaling rules
        self.scaling_rules: List[ScalingRule] = []
        self.current_instances = 1
        self.target_instances = 1
        
        # Cooldown tracking
        self.last_scaling_action: Optional[datetime] = None
        self.scaling_actions_history: List[ScalingAction] = []
        
        # Metrics
        self.current_metrics: Optional[ScalingMetrics] = None
        
        # Initialize default rules
        self._initialize_default_rules()
        
    def _initialize_default_rules(self):
        """Initialize default scaling rules."""
        self.scaling_rules = [
            ScalingRule(
                name="cpu_based",
                trigger=ScalingTrigger.CPU_UTILIZATION,
                scale_up_threshold=70.0,
                scale_down_threshold=30.0,
                min_instances=1,
                max_instances=10,
                cooldown_minutes=5,
                weight=1.0
            ),
            ScalingRule(
                name="memory_based",
                trigger=ScalingTrigger.MEMORY_UTILIZATION,
                scale_up_threshold=80.0,
                scale_down_threshold=40.0,
                min_instances=1,
                max_instances=10,
                cooldown_minutes=3,
                weight=0.8
            ),
            ScalingRule(
                name="response_time_based",
                trigger=ScalingTrigger.RESPONSE_TIME,
                scale_up_threshold=1000.0,  # 1 second
                scale_down_threshold=200.0,  # 200ms
                min_instances=1,
                max_instances=10,
                cooldown_minutes=2,
                weight=0.9
            ),
            ScalingRule(
                name="queue_depth_based",
                trigger=ScalingTrigger.QUEUE_DEPTH,
                scale_up_threshold=50,
                scale_down_threshold=10,
                min_instances=1,
                max_instances=10,
                cooldown_minutes=1,
                weight=0.7
            )
        ]
        
        # Add carbon-aware rule if enabled
        if self.enable_carbon_aware:
            self.scaling_rules.append(
                ScalingRule(
                    name="carbon_aware",
                    trigger=ScalingTrigger.CARBON_INTENSITY,
                    scale_up_threshold=50.0,   # Low carbon - scale up
                    scale_down_threshold=300.0,  # High carbon - scale down
                    min_instances=1,
                    max_instances=10,
                    cooldown_minutes=10,  # Longer cooldown for carbon decisions
                    weight=self.carbon_weight
                )
            )
            
    async def update_metrics(self, metrics: ScalingMetrics) -> None:
        """Update current metrics and trigger scaling evaluation."""
        self.current_metrics = metrics
        
        # Add to predictive scaler
        if self.predictive_scaler:
            self.predictive_scaler.add_metrics(metrics)
            
        # Evaluate scaling need
        scaling_decision = await self._evaluate_scaling()
        
        if scaling_decision.direction != ScalingDirection.STABLE:
            logger.info(f"Scaling decision: {scaling_decision.direction.value} to {scaling_decision.target_instances} instances")
            logger.info(f"Rationale: {scaling_decision.rationale}")
            
            # Record scaling action
            self.scaling_actions_history.append({
                "timestamp": datetime.now(),
                "action": scaling_decision,
                "metrics": metrics
            })
            
    async def _evaluate_scaling(self) -> ScalingAction:
        """Evaluate whether scaling is needed."""
        if not self.current_metrics:
            return ScalingAction(
                direction=ScalingDirection.STABLE,
                target_instances=self.current_instances,
                current_instances=self.current_instances,
                triggered_by=[],
                confidence=0.0,
                estimated_impact={},
                rationale="No metrics available"
            )
            
        # Check cooldown
        if self._in_cooldown():
            return ScalingAction(
                direction=ScalingDirection.STABLE,
                target_instances=self.current_instances,
                current_instances=self.current_instances,
                triggered_by=[],
                confidence=0.0,
                estimated_impact={},
                rationale="In cooldown period"
            )
            
        # Evaluate each scaling rule
        scale_up_votes = []
        scale_down_votes = []
        triggered_by = []
        
        for rule in self.scaling_rules:
            if not rule.enabled:
                continue
                
            vote = await self._evaluate_rule(rule)
            
            if vote['direction'] == ScalingDirection.UP:
                scale_up_votes.append(vote)
                triggered_by.append(rule.name)
            elif vote['direction'] == ScalingDirection.DOWN:
                scale_down_votes.append(vote)
                triggered_by.append(rule.name)
                
        # Carbon-aware scaling override
        carbon_decision = None
        if (self.enable_carbon_aware and 
            self.current_metrics.carbon_intensity is not None):
            should_scale, direction, reason = self.carbon_scaler.should_scale_for_carbon(
                self.current_metrics.carbon_intensity,
                self.current_instances,
                max(rule.max_instances for rule in self.scaling_rules)
            )
            
            if should_scale:
                carbon_decision = {
                    'direction': direction,
                    'weight': self.carbon_weight * 2,  # Higher weight for carbon decisions
                    'reason': reason
                }
                
        # Make scaling decision
        return await self._make_scaling_decision(
            scale_up_votes, 
            scale_down_votes, 
            triggered_by,
            carbon_decision
        )
        
    async def _evaluate_rule(self, rule: ScalingRule) -> Dict[str, Any]:
        """Evaluate a single scaling rule."""
        metrics = self.current_metrics
        current_value = 0.0
        
        # Get metric value based on trigger type
        if rule.trigger == ScalingTrigger.CPU_UTILIZATION:
            current_value = metrics.cpu_utilization
        elif rule.trigger == ScalingTrigger.MEMORY_UTILIZATION:
            current_value = metrics.memory_utilization
        elif rule.trigger == ScalingTrigger.QUEUE_DEPTH:
            current_value = metrics.queue_depth
        elif rule.trigger == ScalingTrigger.RESPONSE_TIME:
            current_value = metrics.response_time_p95
        elif rule.trigger == ScalingTrigger.CARBON_INTENSITY:
            current_value = metrics.carbon_intensity or 0
            
        # Evaluate thresholds
        if current_value >= rule.scale_up_threshold:
            return {
                'direction': ScalingDirection.UP,
                'weight': rule.weight,
                'confidence': min(1.0, (current_value - rule.scale_up_threshold) / rule.scale_up_threshold),
                'current_value': current_value,
                'threshold': rule.scale_up_threshold
            }
        elif current_value <= rule.scale_down_threshold:
            return {
                'direction': ScalingDirection.DOWN,
                'weight': rule.weight,
                'confidence': min(1.0, (rule.scale_down_threshold - current_value) / rule.scale_down_threshold),
                'current_value': current_value,
                'threshold': rule.scale_down_threshold
            }
        else:
            return {
                'direction': ScalingDirection.STABLE,
                'weight': 0,
                'confidence': 0,
                'current_value': current_value,
                'threshold': None
            }
            
    async def _make_scaling_decision(
        self,
        scale_up_votes: List[Dict[str, Any]],
        scale_down_votes: List[Dict[str, Any]],
        triggered_by: List[str],
        carbon_decision: Optional[Dict[str, Any]] = None
    ) -> ScalingAction:
        """Make final scaling decision based on votes."""
        
        # Calculate weighted scores
        up_score = sum(vote['weight'] * vote['confidence'] for vote in scale_up_votes)
        down_score = sum(vote['weight'] * vote['confidence'] for vote in scale_down_votes)
        
        # Apply carbon decision
        if carbon_decision:
            if carbon_decision['direction'] == ScalingDirection.UP:
                up_score += carbon_decision['weight']
            elif carbon_decision['direction'] == ScalingDirection.DOWN:
                down_score += carbon_decision['weight']
            triggered_by.append("carbon_aware")
            
        # Apply predictive scaling
        if self.predictive_scaler:
            prediction = self.predictive_scaler.predict_demand(30)
            if prediction is not None:
                if prediction > 80:
                    up_score += 0.3
                    triggered_by.append("predictive_high")
                elif prediction < 20:
                    down_score += 0.2
                    triggered_by.append("predictive_low")
                    
        # Determine action
        if up_score > down_score and up_score > 0.5:
            direction = ScalingDirection.UP
            target = min(
                self.current_instances + 1,
                max(rule.max_instances for rule in self.scaling_rules)
            )
            confidence = min(1.0, up_score)
            rationale = f"Scale up triggered by {', '.join(triggered_by)} (score: {up_score:.2f})"
            
        elif down_score > up_score and down_score > 0.5:
            direction = ScalingDirection.DOWN
            target = max(
                self.current_instances - 1,
                min(rule.min_instances for rule in self.scaling_rules)
            )
            confidence = min(1.0, down_score)
            rationale = f"Scale down triggered by {', '.join(triggered_by)} (score: {down_score:.2f})"
            
        else:
            direction = ScalingDirection.STABLE
            target = self.current_instances
            confidence = 0.0
            rationale = "No scaling needed (up: {:.2f}, down: {:.2f})".format(up_score, down_score)
            
        # Estimate impact
        estimated_impact = await self._estimate_scaling_impact(target)
        
        return ScalingAction(
            direction=direction,
            target_instances=target,
            current_instances=self.current_instances,
            triggered_by=triggered_by,
            confidence=confidence,
            estimated_impact=estimated_impact,
            rationale=rationale,
            carbon_aware=carbon_decision is not None
        )
        
    async def _estimate_scaling_impact(self, target_instances: int) -> Dict[str, float]:
        """Estimate impact of scaling action."""
        if target_instances == self.current_instances:
            return {}
            
        scale_factor = target_instances / self.current_instances
        
        return {
            "cpu_change_pct": (1 / scale_factor - 1) * 100,  # Negative = reduction
            "memory_change_pct": (1 / scale_factor - 1) * 100,
            "cost_change_pct": (scale_factor - 1) * 100,  # Positive = increase
            "carbon_change_pct": (scale_factor - 1) * 100,
            "latency_change_pct": (1 / scale_factor - 1) * 50,  # Rough estimate
            "throughput_change_pct": (scale_factor - 1) * 80
        }
        
    def _in_cooldown(self) -> bool:
        """Check if scaling is in cooldown period."""
        if not self.last_scaling_action:
            return False
            
        # Use minimum cooldown from triggered rules
        if not self.scaling_actions_history:
            return False
            
        last_action = self.scaling_actions_history[-1]
        min_cooldown = 5  # Default 5 minutes
        
        for rule_name in last_action['action'].triggered_by:
            for rule in self.scaling_rules:
                if rule.name == rule_name:
                    min_cooldown = min(min_cooldown, rule.cooldown_minutes)
                    
        elapsed = (datetime.now() - self.last_scaling_action).total_seconds() / 60
        return elapsed < min_cooldown
        
    async def execute_scaling(self, action: ScalingAction) -> bool:
        """Execute scaling action."""
        if action.direction == ScalingDirection.STABLE:
            return True
            
        logger.info(f"Executing scaling: {self.current_instances} -> {action.target_instances}")
        
        # Update instance count
        self.current_instances = action.target_instances
        self.target_instances = action.target_instances
        self.last_scaling_action = datetime.now()
        
        # In a real implementation, this would trigger actual scaling
        # (e.g., Kubernetes HPA, AWS Auto Scaling, etc.)
        
        return True
        
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        recent_actions = self.scaling_actions_history[-5:] if self.scaling_actions_history else []
        
        patterns = {}
        if self.predictive_scaler:
            patterns = self.predictive_scaler.detect_patterns()
            
        carbon_efficiency = 0.5
        if self.carbon_scaler:
            carbon_efficiency = self.carbon_scaler.get_carbon_efficiency_score(self.current_instances)
            
        return {
            "current_instances": self.current_instances,
            "target_instances": self.target_instances,
            "in_cooldown": self._in_cooldown(),
            "last_scaling_action": self.last_scaling_action.isoformat() if self.last_scaling_action else None,
            "total_scaling_actions": len(self.scaling_actions_history),
            "recent_actions": [
                {
                    "timestamp": action["timestamp"].isoformat(),
                    "direction": action["action"].direction.value,
                    "target": action["action"].target_instances,
                    "triggered_by": action["action"].triggered_by,
                    "confidence": action["action"].confidence
                }
                for action in recent_actions
            ],
            "usage_patterns": patterns,
            "carbon_efficiency_score": carbon_efficiency,
            "enabled_features": {
                "predictive": self.enable_predictive,
                "carbon_aware": self.enable_carbon_aware
            }
        }
        
    def add_scaling_rule(self, rule: ScalingRule) -> None:
        """Add custom scaling rule."""
        self.scaling_rules.append(rule)
        logger.info(f"Added scaling rule: {rule.name}")
        
    def remove_scaling_rule(self, rule_name: str) -> bool:
        """Remove scaling rule by name."""
        for i, rule in enumerate(self.scaling_rules):
            if rule.name == rule_name:
                del self.scaling_rules[i]
                logger.info(f"Removed scaling rule: {rule_name}")
                return True
        return False


# Global intelligent auto-scaler
intelligent_scaler = IntelligentAutoScaler()