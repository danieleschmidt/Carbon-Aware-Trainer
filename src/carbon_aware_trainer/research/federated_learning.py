"""
Federated Carbon-Aware Learning Framework.

This module implements privacy-preserving federated learning for carbon-aware
ML training, enabling organizations to share carbon optimization insights
without exposing sensitive data.
"""

import asyncio
import logging
import math
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any, Callable
from enum import Enum
import json
from pathlib import Path

# Optional cryptographic and numerical libraries
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

from ..core.types import CarbonIntensity, CarbonForecast, TrainingMetrics
from ..core.monitor import CarbonMonitor
from ..core.advanced_forecasting import AdvancedCarbonForecaster, MultiModalInputs


logger = logging.getLogger(__name__)


class FederatedRole(Enum):
    """Roles in federated carbon-aware learning."""
    COORDINATOR = "coordinator"  # Central coordinator
    PARTICIPANT = "participant"  # Data contributor
    AGGREGATOR = "aggregator"    # Model aggregation service
    VALIDATOR = "validator"      # Result validation service


class PrivacyMechanism(Enum):
    """Privacy preservation mechanisms."""
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    SECURE_AGGREGATION = "secure_aggregation"
    HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"
    LOCAL_DIFFERENTIAL_PRIVACY = "local_differential_privacy"


@dataclass
class FederatedParticipant:
    """Participant in federated carbon learning."""
    participant_id: str
    region: str
    organization: str
    trust_score: float = 1.0
    data_quality_score: float = 1.0
    contribution_weight: float = 1.0
    privacy_level: float = 1.0  # Epsilon for differential privacy
    last_update: Optional[datetime] = None


@dataclass
class PrivacyParameters:
    """Privacy parameters for federated learning."""
    epsilon: float = 1.0  # Differential privacy budget
    delta: float = 1e-6   # DP delta parameter
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    secure_aggregation: bool = True


@dataclass
class CarbonPattern:
    """Carbon intensity pattern for sharing."""
    pattern_id: str
    region: str
    temporal_signature: List[float]  # Normalized patterns
    seasonal_components: Dict[str, List[float]]
    renewable_correlation: float
    confidence_score: float
    privacy_noise_level: float


@dataclass
class FederatedUpdate:
    """Update from federated participant."""
    participant_id: str
    update_timestamp: datetime
    model_weights: Dict[str, List[float]]
    local_loss: float
    data_size: int
    privacy_spent: float
    validation_metrics: Dict[str, float]


class DifferentialPrivacy:
    """Differential privacy implementation for federated learning."""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-6):
        """Initialize differential privacy mechanism.
        
        Args:
            epsilon: Privacy budget
            delta: DP delta parameter
        """
        self.epsilon = epsilon
        self.delta = delta
        
    def add_gaussian_noise(self, values: List[float], sensitivity: float = 1.0) -> List[float]:
        """Add Gaussian noise for differential privacy."""
        import random
        
        # Calculate noise scale for Gaussian mechanism
        noise_scale = (sensitivity * math.sqrt(2 * math.log(1.25 / self.delta))) / self.epsilon
        
        noisy_values = []
        for value in values:
            noise = random.gauss(0, noise_scale)
            noisy_values.append(value + noise)
        
        return noisy_values
    
    def add_laplace_noise(self, values: List[float], sensitivity: float = 1.0) -> List[float]:
        """Add Laplace noise for differential privacy."""
        import random
        
        # Laplace mechanism
        scale = sensitivity / self.epsilon
        
        noisy_values = []
        for value in values:
            # Generate Laplace noise
            u = random.uniform(-0.5, 0.5)
            noise = -scale * math.copysign(math.log(1 - 2 * abs(u)), u)
            noisy_values.append(value + noise)
        
        return noisy_values
    
    def clip_and_add_noise(self, gradients: Dict[str, List[float]], max_norm: float = 1.0) -> Dict[str, List[float]]:
        """Clip gradients and add noise for privacy."""
        clipped_gradients = {}
        
        for param_name, grad_values in gradients.items():
            # Compute L2 norm
            l2_norm = math.sqrt(sum(g * g for g in grad_values))
            
            # Clip gradients if necessary
            if l2_norm > max_norm:
                clip_factor = max_norm / l2_norm
                clipped_values = [g * clip_factor for g in grad_values]
            else:
                clipped_values = grad_values[:]
            
            # Add noise
            noisy_values = self.add_gaussian_noise(clipped_values, max_norm)
            clipped_gradients[param_name] = noisy_values
        
        return clipped_gradients


class SecureAggregation:
    """Secure aggregation protocol for federated learning."""
    
    def __init__(self):
        """Initialize secure aggregation."""
        self.participant_secrets = {}
        self.aggregation_masks = {}
    
    def generate_secret_shares(self, participant_id: str, num_participants: int) -> Dict[str, float]:
        """Generate secret shares for secure aggregation."""
        import random
        
        # Generate random secret
        secret = random.uniform(-1000, 1000)
        self.participant_secrets[participant_id] = secret
        
        # Generate shares that sum to zero (except for dropout resilience)
        shares = {}
        remaining_secret = secret
        
        for i in range(num_participants - 1):
            share_id = f"share_{i}"
            share_value = random.uniform(-secret, secret)
            shares[share_id] = share_value
            remaining_secret -= share_value
        
        shares[f"share_{num_participants - 1}"] = remaining_secret
        return shares
    
    def create_aggregation_mask(self, participant_weights: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """Create aggregation mask for secure computation."""
        import random
        
        masked_weights = {}
        
        for param_name, weights in participant_weights.items():
            # Create random mask
            mask = [random.uniform(-10, 10) for _ in weights]
            self.aggregation_masks[param_name] = mask
            
            # Apply mask
            masked_weights[param_name] = [w + m for w, m in zip(weights, mask)]
        
        return masked_weights
    
    def remove_aggregation_mask(self, aggregated_weights: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """Remove aggregation mask after secure aggregation."""
        unmasked_weights = {}
        
        for param_name, weights in aggregated_weights.items():
            if param_name in self.aggregation_masks:
                mask = self.aggregation_masks[param_name]
                unmasked_weights[param_name] = [w - m for w, m in zip(weights, mask)]
            else:
                unmasked_weights[param_name] = weights
        
        return unmasked_weights


class FederatedCarbonLearning:
    """
    Federated learning system for carbon-aware ML training.
    
    Enables multiple organizations to collaboratively learn carbon
    optimization patterns while preserving data privacy.
    """
    
    def __init__(
        self,
        participant_id: str,
        role: FederatedRole,
        privacy_params: PrivacyParameters = None
    ):
        """Initialize federated carbon learning participant.
        
        Args:
            participant_id: Unique identifier for this participant
            role: Role in the federated system
            privacy_params: Privacy preservation parameters
        """
        self.participant_id = participant_id
        self.role = role
        self.privacy_params = privacy_params or PrivacyParameters()
        
        # Privacy mechanisms
        self.differential_privacy = DifferentialPrivacy(
            self.privacy_params.epsilon,
            self.privacy_params.delta
        )
        self.secure_aggregation = SecureAggregation()
        
        # Participant management
        self.participants: Dict[str, FederatedParticipant] = {}
        self.carbon_patterns: Dict[str, CarbonPattern] = {}
        self.federated_updates: List[FederatedUpdate] = []
        self.global_model_weights: Dict[str, List[float]] = {}
        
        # Learning state
        self.round_number = 0
        self.privacy_budget_spent = 0.0
        self.contribution_history = []
        
        logger.info(f"Initialized federated carbon learning: {participant_id} ({role.value})")
    
    async def register_participant(self, participant: FederatedParticipant) -> bool:
        """Register a new participant in the federated system."""
        if participant.participant_id in self.participants:
            logger.warning(f"Participant {participant.participant_id} already registered")
            return False
        
        # Validate participant credentials (simplified)
        if self._validate_participant(participant):
            self.participants[participant.participant_id] = participant
            logger.info(f"Registered participant: {participant.participant_id} from {participant.region}")
            return True
        else:
            logger.warning(f"Failed to validate participant: {participant.participant_id}")
            return False
    
    def _validate_participant(self, participant: FederatedParticipant) -> bool:
        """Validate participant credentials and trustworthiness."""
        # Basic validation (would include cryptographic verification in practice)
        if not participant.participant_id or not participant.region:
            return False
        
        if participant.trust_score < 0.1:  # Minimum trust threshold
            return False
        
        if participant.data_quality_score < 0.1:  # Minimum data quality
            return False
        
        return True
    
    async def extract_local_carbon_patterns(
        self,
        carbon_data: List[CarbonIntensity],
        forecaster: AdvancedCarbonForecaster
    ) -> List[CarbonPattern]:
        """Extract local carbon patterns for federated sharing."""
        logger.info(f"Extracting carbon patterns from {len(carbon_data)} data points")
        
        patterns = []
        
        try:
            # Extract daily patterns
            daily_pattern = await self._extract_daily_pattern(carbon_data)
            if daily_pattern:
                patterns.append(daily_pattern)
            
            # Extract weekly patterns
            weekly_pattern = await self._extract_weekly_pattern(carbon_data)
            if weekly_pattern:
                patterns.append(weekly_pattern)
            
            # Extract seasonal patterns
            seasonal_pattern = await self._extract_seasonal_pattern(carbon_data)
            if seasonal_pattern:
                patterns.append(seasonal_pattern)
            
            # Extract renewable correlation patterns
            renewable_pattern = await self._extract_renewable_pattern(carbon_data)
            if renewable_pattern:
                patterns.append(renewable_pattern)
        
        except Exception as e:
            logger.error(f"Failed to extract carbon patterns: {e}")
        
        logger.info(f"Extracted {len(patterns)} carbon patterns")
        return patterns
    
    async def _extract_daily_pattern(self, carbon_data: List[CarbonIntensity]) -> Optional[CarbonPattern]:
        """Extract daily carbon intensity patterns."""
        if len(carbon_data) < 48:  # Need at least 2 days
            return None
        
        # Group by hour of day
        hourly_intensities = [[] for _ in range(24)]
        
        for ci in carbon_data:
            hour = ci.timestamp.hour
            hourly_intensities[hour].append(ci.carbon_intensity)
        
        # Calculate average intensity for each hour
        daily_signature = []
        for hour_data in hourly_intensities:
            if hour_data:
                avg_intensity = sum(hour_data) / len(hour_data)
                # Normalize to 0-1 range
                normalized_intensity = min(1.0, max(0.0, (avg_intensity - 50) / 300))
                daily_signature.append(normalized_intensity)
            else:
                daily_signature.append(0.5)  # Default neutral value
        
        # Add privacy noise
        noisy_signature = self.differential_privacy.add_gaussian_noise(
            daily_signature, sensitivity=0.1
        )
        
        # Calculate confidence based on data availability
        total_data_points = sum(len(hour_data) for hour_data in hourly_intensities)
        confidence = min(1.0, total_data_points / (24 * 7))  # Full confidence with 1 week of data
        
        pattern = CarbonPattern(
            pattern_id=f"{self.participant_id}_daily_{datetime.now().strftime('%Y%m%d')}",
            region=carbon_data[0].region if carbon_data else "unknown",
            temporal_signature=noisy_signature,
            seasonal_components={"daily": noisy_signature},
            renewable_correlation=0.0,  # Would calculate separately
            confidence_score=confidence,
            privacy_noise_level=self.privacy_params.epsilon
        )
        
        return pattern
    
    async def _extract_weekly_pattern(self, carbon_data: List[CarbonIntensity]) -> Optional[CarbonPattern]:
        """Extract weekly carbon intensity patterns."""
        if len(carbon_data) < 168:  # Need at least 1 week
            return None
        
        # Group by day of week
        daily_intensities = [[] for _ in range(7)]
        
        for ci in carbon_data:
            day_of_week = ci.timestamp.weekday()
            daily_intensities[day_of_week].append(ci.carbon_intensity)
        
        # Calculate average intensity for each day
        weekly_signature = []
        for day_data in daily_intensities:
            if day_data:
                avg_intensity = sum(day_data) / len(day_data)
                normalized_intensity = min(1.0, max(0.0, (avg_intensity - 50) / 300))
                weekly_signature.append(normalized_intensity)
            else:
                weekly_signature.append(0.5)
        
        # Add privacy noise
        noisy_signature = self.differential_privacy.add_gaussian_noise(
            weekly_signature, sensitivity=0.1
        )
        
        confidence = min(1.0, len(carbon_data) / (168 * 4))  # Full confidence with 1 month of data
        
        pattern = CarbonPattern(
            pattern_id=f"{self.participant_id}_weekly_{datetime.now().strftime('%Y%m%d')}",
            region=carbon_data[0].region if carbon_data else "unknown",
            temporal_signature=noisy_signature,
            seasonal_components={"weekly": noisy_signature},
            renewable_correlation=0.0,
            confidence_score=confidence,
            privacy_noise_level=self.privacy_params.epsilon
        )
        
        return pattern
    
    async def _extract_seasonal_pattern(self, carbon_data: List[CarbonIntensity]) -> Optional[CarbonPattern]:
        """Extract seasonal carbon intensity patterns."""
        if len(carbon_data) < 720:  # Need at least 1 month
            return None
        
        # Group by month
        monthly_intensities = [[] for _ in range(12)]
        
        for ci in carbon_data:
            month = ci.timestamp.month - 1  # 0-indexed
            monthly_intensities[month].append(ci.carbon_intensity)
        
        # Calculate seasonal signature
        seasonal_signature = []
        for month_data in monthly_intensities:
            if month_data:
                avg_intensity = sum(month_data) / len(month_data)
                normalized_intensity = min(1.0, max(0.0, (avg_intensity - 50) / 300))
                seasonal_signature.append(normalized_intensity)
            else:
                seasonal_signature.append(0.5)
        
        # Add privacy noise
        noisy_signature = self.differential_privacy.add_gaussian_noise(
            seasonal_signature, sensitivity=0.1
        )
        
        confidence = min(1.0, len(carbon_data) / (8760))  # Full confidence with 1 year of data
        
        pattern = CarbonPattern(
            pattern_id=f"{self.participant_id}_seasonal_{datetime.now().strftime('%Y%m%d')}",
            region=carbon_data[0].region if carbon_data else "unknown",
            temporal_signature=noisy_signature,
            seasonal_components={"seasonal": noisy_signature},
            renewable_correlation=0.0,
            confidence_score=confidence,
            privacy_noise_level=self.privacy_params.epsilon
        )
        
        return pattern
    
    async def _extract_renewable_pattern(self, carbon_data: List[CarbonIntensity]) -> Optional[CarbonPattern]:
        """Extract renewable energy correlation patterns."""
        # Filter data with renewable percentage information
        renewable_data = [ci for ci in carbon_data if ci.renewable_percentage is not None]
        
        if len(renewable_data) < 100:
            return None
        
        # Calculate correlation between renewable percentage and carbon intensity
        renewable_percentages = [ci.renewable_percentage for ci in renewable_data]
        carbon_intensities = [ci.carbon_intensity for ci in renewable_data]
        
        # Simple correlation calculation
        n = len(renewable_percentages)
        sum_renewable = sum(renewable_percentages)
        sum_carbon = sum(carbon_intensities)
        sum_renewable_carbon = sum(r * c for r, c in zip(renewable_percentages, carbon_intensities))
        sum_renewable_sq = sum(r * r for r in renewable_percentages)
        sum_carbon_sq = sum(c * c for c in carbon_intensities)
        
        numerator = n * sum_renewable_carbon - sum_renewable * sum_carbon
        denominator = math.sqrt((n * sum_renewable_sq - sum_renewable ** 2) * (n * sum_carbon_sq - sum_carbon ** 2))
        
        correlation = numerator / denominator if denominator != 0 else 0.0
        
        # Add privacy noise to correlation
        noisy_correlation = self.differential_privacy.add_gaussian_noise([correlation], sensitivity=0.1)[0]
        
        # Create temporal signature based on renewable percentages
        temporal_signature = renewable_percentages[:24] if len(renewable_percentages) >= 24 else renewable_percentages
        # Normalize to 0-1 range
        temporal_signature = [min(1.0, max(0.0, r)) for r in temporal_signature]
        
        # Add noise to signature
        noisy_signature = self.differential_privacy.add_gaussian_noise(temporal_signature, sensitivity=0.05)
        
        pattern = CarbonPattern(
            pattern_id=f"{self.participant_id}_renewable_{datetime.now().strftime('%Y%m%d')}",
            region=renewable_data[0].region if renewable_data else "unknown",
            temporal_signature=noisy_signature,
            seasonal_components={"renewable": noisy_signature},
            renewable_correlation=noisy_correlation,
            confidence_score=min(1.0, len(renewable_data) / 1000),
            privacy_noise_level=self.privacy_params.epsilon
        )
        
        return pattern
    
    async def share_patterns_privately(self, patterns: List[CarbonPattern]) -> Dict[str, Any]:
        """Share carbon patterns with privacy preservation."""
        if self.role != FederatedRole.PARTICIPANT:
            raise ValueError("Only participants can share patterns")
        
        logger.info(f"Sharing {len(patterns)} carbon patterns privately")
        
        shared_data = {
            "participant_id": self.participant_id,
            "timestamp": datetime.now().isoformat(),
            "patterns": [],
            "privacy_metadata": {
                "epsilon": self.privacy_params.epsilon,
                "delta": self.privacy_params.delta,
                "noise_mechanism": "gaussian"
            }
        }
        
        for pattern in patterns:
            # Additional privacy processing if needed
            pattern_data = {
                "pattern_id": pattern.pattern_id,
                "region": pattern.region,
                "temporal_signature": pattern.temporal_signature,
                "seasonal_components": pattern.seasonal_components,
                "renewable_correlation": pattern.renewable_correlation,
                "confidence_score": pattern.confidence_score,
                "privacy_noise_level": pattern.privacy_noise_level
            }
            shared_data["patterns"].append(pattern_data)
            
            # Store locally
            self.carbon_patterns[pattern.pattern_id] = pattern
        
        # Update privacy budget
        privacy_cost = len(patterns) * self.privacy_params.epsilon
        self.privacy_budget_spent += privacy_cost
        
        logger.info(f"Shared patterns with privacy cost: {privacy_cost:.3f} (total spent: {self.privacy_budget_spent:.3f})")
        
        return shared_data
    
    async def aggregate_federated_patterns(self, participant_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate carbon patterns from multiple participants."""
        if self.role != FederatedRole.COORDINATOR and self.role != FederatedRole.AGGREGATOR:
            raise ValueError("Only coordinators/aggregators can aggregate patterns")
        
        logger.info(f"Aggregating patterns from {len(participant_patterns)} participants")
        
        # Group patterns by type (daily, weekly, seasonal, renewable)
        pattern_groups = {
            "daily": [],
            "weekly": [],
            "seasonal": [],
            "renewable": []
        }
        
        for participant_data in participant_patterns:
            for pattern_data in participant_data.get("patterns", []):
                # Determine pattern type from seasonal components
                for component_type in pattern_data.get("seasonal_components", {}):
                    if component_type in pattern_groups:
                        pattern_groups[component_type].append(pattern_data)
        
        # Aggregate each pattern type
        aggregated_patterns = {}
        
        for pattern_type, patterns in pattern_groups.items():
            if not patterns:
                continue
            
            # Weighted aggregation based on confidence scores
            total_weight = sum(p.get("confidence_score", 0.5) for p in patterns)
            
            if total_weight == 0:
                continue
            
            # Aggregate temporal signatures
            signature_length = len(patterns[0].get("temporal_signature", []))
            aggregated_signature = []
            
            for i in range(signature_length):
                weighted_sum = sum(
                    p.get("temporal_signature", [0])[i] * p.get("confidence_score", 0.5)
                    for p in patterns if len(p.get("temporal_signature", [])) > i
                )
                aggregated_signature.append(weighted_sum / total_weight if total_weight > 0 else 0.0)
            
            # Aggregate renewable correlations
            renewable_correlations = [p.get("renewable_correlation", 0.0) for p in patterns]
            avg_renewable_correlation = sum(renewable_correlations) / len(renewable_correlations) if renewable_correlations else 0.0
            
            aggregated_patterns[pattern_type] = {
                "temporal_signature": aggregated_signature,
                "renewable_correlation": avg_renewable_correlation,
                "confidence_score": total_weight / len(patterns),
                "participant_count": len(patterns),
                "regions_represented": list(set(p.get("region", "unknown") for p in patterns))
            }
        
        federated_result = {
            "aggregation_timestamp": datetime.now().isoformat(),
            "round_number": self.round_number,
            "aggregated_patterns": aggregated_patterns,
            "total_participants": len(participant_patterns),
            "privacy_preserved": True,
            "aggregation_method": "weighted_confidence_based"
        }
        
        self.round_number += 1
        logger.info(f"Aggregated {len(aggregated_patterns)} pattern types from {len(participant_patterns)} participants")
        
        return federated_result
    
    async def apply_federated_insights(
        self,
        federated_patterns: Dict[str, Any],
        local_forecaster: AdvancedCarbonForecaster
    ) -> Dict[str, Any]:
        """Apply federated insights to improve local carbon forecasting."""
        logger.info("Applying federated insights to local forecaster")
        
        insights_applied = {
            "application_timestamp": datetime.now().isoformat(),
            "insights_applied": [],
            "improvement_estimates": {},
            "confidence_boost": 0.0
        }
        
        aggregated_patterns = federated_patterns.get("aggregated_patterns", {})
        
        for pattern_type, pattern_data in aggregated_patterns.items():
            try:
                # Apply pattern-specific insights
                if pattern_type == "daily" and "temporal_signature" in pattern_data:
                    # Use federated daily pattern to improve local forecasting
                    daily_insights = self._apply_daily_insights(pattern_data, local_forecaster)
                    insights_applied["insights_applied"].append({
                        "type": "daily_pattern",
                        "confidence_improvement": daily_insights.get("confidence_boost", 0.0),
                        "regions_learned_from": pattern_data.get("regions_represented", [])
                    })
                
                elif pattern_type == "weekly" and "temporal_signature" in pattern_data:
                    weekly_insights = self._apply_weekly_insights(pattern_data, local_forecaster)
                    insights_applied["insights_applied"].append({
                        "type": "weekly_pattern",
                        "confidence_improvement": weekly_insights.get("confidence_boost", 0.0),
                        "regions_learned_from": pattern_data.get("regions_represented", [])
                    })
                
                elif pattern_type == "renewable" and "renewable_correlation" in pattern_data:
                    renewable_insights = self._apply_renewable_insights(pattern_data, local_forecaster)
                    insights_applied["insights_applied"].append({
                        "type": "renewable_correlation",
                        "correlation_learned": pattern_data["renewable_correlation"],
                        "regions_learned_from": pattern_data.get("regions_represented", [])
                    })
            
            except Exception as e:
                logger.warning(f"Failed to apply {pattern_type} insights: {e}")
        
        # Calculate overall confidence boost
        confidence_boosts = [insight.get("confidence_improvement", 0.0) for insight in insights_applied["insights_applied"]]
        insights_applied["confidence_boost"] = sum(confidence_boosts) / len(confidence_boosts) if confidence_boosts else 0.0
        
        # Estimate improvement in forecasting accuracy
        participant_count = federated_patterns.get("total_participants", 1)
        diversity_bonus = min(0.2, participant_count * 0.02)  # Up to 20% improvement from diversity
        
        insights_applied["improvement_estimates"] = {
            "accuracy_improvement_estimate": diversity_bonus + insights_applied["confidence_boost"] * 0.1,
            "uncertainty_reduction_estimate": insights_applied["confidence_boost"] * 0.05,
            "cross_regional_knowledge_gain": len(set().union(*[insight.get("regions_learned_from", []) for insight in insights_applied["insights_applied"]]))
        }
        
        logger.info(f"Applied federated insights: {insights_applied['confidence_boost']:.3f} confidence boost, {len(insights_applied['insights_applied'])} insights")
        
        return insights_applied
    
    def _apply_daily_insights(self, daily_pattern: Dict[str, Any], forecaster: AdvancedCarbonForecaster) -> Dict[str, Any]:
        """Apply daily pattern insights to local forecaster."""
        # In a real implementation, this would update the forecaster's daily pattern model
        temporal_signature = daily_pattern.get("temporal_signature", [])
        confidence_score = daily_pattern.get("confidence_score", 0.5)
        
        # Simulate confidence boost from federated learning
        confidence_boost = min(0.3, confidence_score * 0.5)  # Up to 30% boost
        
        logger.debug(f"Applied daily pattern insights with {confidence_boost:.3f} confidence boost")
        
        return {
            "confidence_boost": confidence_boost,
            "pattern_length": len(temporal_signature),
            "federated_confidence": confidence_score
        }
    
    def _apply_weekly_insights(self, weekly_pattern: Dict[str, Any], forecaster: AdvancedCarbonForecaster) -> Dict[str, Any]:
        """Apply weekly pattern insights to local forecaster."""
        temporal_signature = weekly_pattern.get("temporal_signature", [])
        confidence_score = weekly_pattern.get("confidence_score", 0.5)
        
        confidence_boost = min(0.25, confidence_score * 0.4)  # Up to 25% boost
        
        logger.debug(f"Applied weekly pattern insights with {confidence_boost:.3f} confidence boost")
        
        return {
            "confidence_boost": confidence_boost,
            "pattern_length": len(temporal_signature),
            "federated_confidence": confidence_score
        }
    
    def _apply_renewable_insights(self, renewable_pattern: Dict[str, Any], forecaster: AdvancedCarbonForecaster) -> Dict[str, Any]:
        """Apply renewable energy correlation insights to local forecaster."""
        correlation = renewable_pattern.get("renewable_correlation", 0.0)
        confidence_score = renewable_pattern.get("confidence_score", 0.5)
        
        # Strong correlations provide more insight
        correlation_strength = abs(correlation)
        confidence_boost = min(0.2, correlation_strength * confidence_score * 0.3)
        
        logger.debug(f"Applied renewable correlation insights: r={correlation:.3f}, boost={confidence_boost:.3f}")
        
        return {
            "confidence_boost": confidence_boost,
            "correlation_learned": correlation,
            "correlation_strength": correlation_strength
        }
    
    async def evaluate_federated_performance(
        self,
        baseline_metrics: Dict[str, float],
        federated_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Evaluate the performance impact of federated learning."""
        evaluation = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "baseline_performance": baseline_metrics,
            "federated_performance": federated_metrics,
            "improvements": {},
            "privacy_cost_benefit": {},
            "overall_assessment": ""
        }
        
        # Calculate improvements
        for metric, baseline_value in baseline_metrics.items():
            if metric in federated_metrics:
                federated_value = federated_metrics[metric]
                
                # For error metrics (MAE, RMSE), lower is better
                if metric.lower() in ['mae', 'rmse', 'mape']:
                    improvement = (baseline_value - federated_value) / baseline_value
                    evaluation["improvements"][metric] = {
                        "absolute_improvement": baseline_value - federated_value,
                        "relative_improvement_percent": improvement * 100,
                        "direction": "lower_is_better"
                    }
                else:
                    # For other metrics (R², confidence), higher is better
                    improvement = (federated_value - baseline_value) / baseline_value
                    evaluation["improvements"][metric] = {
                        "absolute_improvement": federated_value - baseline_value,
                        "relative_improvement_percent": improvement * 100,
                        "direction": "higher_is_better"
                    }
        
        # Privacy cost-benefit analysis
        accuracy_improvements = [
            imp["relative_improvement_percent"] for imp in evaluation["improvements"].values()
            if imp["relative_improvement_percent"] > 0
        ]
        
        avg_improvement = sum(accuracy_improvements) / len(accuracy_improvements) if accuracy_improvements else 0.0
        
        evaluation["privacy_cost_benefit"] = {
            "privacy_budget_spent": self.privacy_budget_spent,
            "average_improvement_percent": avg_improvement,
            "improvement_per_epsilon": avg_improvement / max(self.privacy_budget_spent, 0.01),
            "participants_contributed": len(self.participants),
            "patterns_learned": len(self.carbon_patterns)
        }
        
        # Overall assessment
        if avg_improvement > 10.0 and self.privacy_budget_spent < 2.0:
            evaluation["overall_assessment"] = "Excellent: Significant improvement with reasonable privacy cost"
        elif avg_improvement > 5.0 and self.privacy_budget_spent < 3.0:
            evaluation["overall_assessment"] = "Good: Moderate improvement with acceptable privacy cost"
        elif avg_improvement > 2.0:
            evaluation["overall_assessment"] = "Fair: Small improvement, evaluate privacy tradeoffs"
        else:
            evaluation["overall_assessment"] = "Poor: Insufficient improvement to justify privacy cost"
        
        logger.info(f"Federated learning evaluation: {avg_improvement:.1f}% avg improvement, {self.privacy_budget_spent:.2f} privacy budget spent")
        
        return evaluation
    
    def save_federated_state(self, filepath: str) -> None:
        """Save federated learning state to file."""
        state = {
            "participant_id": self.participant_id,
            "role": self.role.value,
            "round_number": self.round_number,
            "privacy_budget_spent": self.privacy_budget_spent,
            "participants": {
                pid: {
                    "participant_id": p.participant_id,
                    "region": p.region,
                    "organization": p.organization,
                    "trust_score": p.trust_score,
                    "data_quality_score": p.data_quality_score,
                    "contribution_weight": p.contribution_weight,
                    "privacy_level": p.privacy_level,
                    "last_update": p.last_update.isoformat() if p.last_update else None
                }
                for pid, p in self.participants.items()
            },
            "carbon_patterns": {
                pid: {
                    "pattern_id": p.pattern_id,
                    "region": p.region,
                    "temporal_signature": p.temporal_signature,
                    "seasonal_components": p.seasonal_components,
                    "renewable_correlation": p.renewable_correlation,
                    "confidence_score": p.confidence_score,
                    "privacy_noise_level": p.privacy_noise_level
                }
                for pid, p in self.carbon_patterns.items()
            },
            "privacy_params": {
                "epsilon": self.privacy_params.epsilon,
                "delta": self.privacy_params.delta,
                "noise_multiplier": self.privacy_params.noise_multiplier,
                "max_grad_norm": self.privacy_params.max_grad_norm,
                "secure_aggregation": self.privacy_params.secure_aggregation
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Saved federated learning state to {filepath}")
    
    def load_federated_state(self, filepath: str) -> None:
        """Load federated learning state from file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.participant_id = state["participant_id"]
        self.role = FederatedRole(state["role"])
        self.round_number = state["round_number"]
        self.privacy_budget_spent = state["privacy_budget_spent"]
        
        # Restore participants
        self.participants = {}
        for pid, p_data in state["participants"].items():
            participant = FederatedParticipant(
                participant_id=p_data["participant_id"],
                region=p_data["region"],
                organization=p_data["organization"],
                trust_score=p_data["trust_score"],
                data_quality_score=p_data["data_quality_score"],
                contribution_weight=p_data["contribution_weight"],
                privacy_level=p_data["privacy_level"],
                last_update=datetime.fromisoformat(p_data["last_update"]) if p_data["last_update"] else None
            )
            self.participants[pid] = participant
        
        # Restore carbon patterns
        self.carbon_patterns = {}
        for pid, p_data in state["carbon_patterns"].items():
            pattern = CarbonPattern(
                pattern_id=p_data["pattern_id"],
                region=p_data["region"],
                temporal_signature=p_data["temporal_signature"],
                seasonal_components=p_data["seasonal_components"],
                renewable_correlation=p_data["renewable_correlation"],
                confidence_score=p_data["confidence_score"],
                privacy_noise_level=p_data["privacy_noise_level"]
            )
            self.carbon_patterns[pid] = pattern
        
        # Restore privacy parameters
        privacy_data = state["privacy_params"]
        self.privacy_params = PrivacyParameters(
            epsilon=privacy_data["epsilon"],
            delta=privacy_data["delta"],
            noise_multiplier=privacy_data["noise_multiplier"],
            max_grad_norm=privacy_data["max_grad_norm"],
            secure_aggregation=privacy_data["secure_aggregation"]
        )
        
        logger.info(f"Loaded federated learning state from {filepath}")


class FederatedCarbonOrchestrator:
    """
    Orchestrator for federated carbon-aware learning across multiple participants.
    
    Coordinates the federated learning process, manages participants,
    and ensures privacy preservation throughout the learning cycle.
    """
    
    def __init__(self, orchestrator_id: str):
        """Initialize federated carbon orchestrator.
        
        Args:
            orchestrator_id: Unique identifier for this orchestrator
        """
        self.orchestrator_id = orchestrator_id
        self.participants: Dict[str, FederatedCarbonLearning] = {}
        self.learning_rounds = []
        self.global_patterns = {}
        
        logger.info(f"Initialized federated carbon orchestrator: {orchestrator_id}")
    
    async def orchestrate_federated_learning(
        self,
        participants: List[FederatedCarbonLearning],
        rounds: int = 5,
        min_participants: int = 3
    ) -> Dict[str, Any]:
        """Orchestrate complete federated learning process."""
        if len(participants) < min_participants:
            raise ValueError(f"Need at least {min_participants} participants for federated learning")
        
        logger.info(f"Starting federated learning with {len(participants)} participants for {rounds} rounds")
        
        orchestration_result = {
            "orchestrator_id": self.orchestrator_id,
            "start_timestamp": datetime.now().isoformat(),
            "participants": [p.participant_id for p in participants],
            "rounds_completed": 0,
            "final_patterns": {},
            "performance_improvements": {},
            "privacy_summary": {}
        }
        
        for round_num in range(rounds):
            logger.info(f"Starting federated learning round {round_num + 1}/{rounds}")
            
            try:
                # Collect patterns from all participants
                round_patterns = []
                for participant in participants:
                    if participant.role == FederatedRole.PARTICIPANT:
                        # In practice, would collect actual carbon data and extract patterns
                        patterns = []  # Placeholder - would use actual data
                        shared_data = await participant.share_patterns_privately(patterns)
                        round_patterns.append(shared_data)
                
                # Aggregate patterns
                coordinator = next((p for p in participants if p.role == FederatedRole.COORDINATOR), None)
                if coordinator:
                    aggregated = await coordinator.aggregate_federated_patterns(round_patterns)
                    orchestration_result["final_patterns"] = aggregated
                
                # Apply insights to all participants
                for participant in participants:
                    if participant.role == FederatedRole.PARTICIPANT:
                        insights = await participant.apply_federated_insights(
                            orchestration_result["final_patterns"],
                            None  # Would pass actual forecaster
                        )
                
                orchestration_result["rounds_completed"] += 1
                
            except Exception as e:
                logger.error(f"Federated learning round {round_num + 1} failed: {e}")
                break
        
        orchestration_result["end_timestamp"] = datetime.now().isoformat()
        
        # Calculate privacy summary
        total_privacy_spent = sum(p.privacy_budget_spent for p in participants)
        orchestration_result["privacy_summary"] = {
            "total_privacy_budget_spent": total_privacy_spent,
            "average_privacy_per_participant": total_privacy_spent / len(participants),
            "privacy_efficiency": orchestration_result["rounds_completed"] / max(total_privacy_spent, 0.01)
        }
        
        logger.info(f"Completed federated learning: {orchestration_result['rounds_completed']} rounds, {total_privacy_spent:.2f} total privacy budget")
        
        return orchestration_result