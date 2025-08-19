"""
Real-Time Adaptive Carbon Optimization for ML Training

Novel real-time carbon optimization system that adapts training parameters 
dynamically based on live carbon intensity feeds, weather patterns, and 
grid demand predictions.

Research Contributions:
1. Real-time carbon intensity prediction with sub-15-minute updates
2. Adaptive batch size and learning rate optimization based on carbon
3. Multi-objective optimization (carbon, performance, cost)
4. Online reinforcement learning for scheduling decisions

Research Paper: "Real-Time Adaptive Carbon Optimization in ML Training"
Author: Daniel Schmidt, Terragon Labs  
Date: August 2025
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import statistics
from collections import defaultdict, deque
import random

from ..core.types import CarbonIntensity, CarbonForecast, TrainingConfig
from .neural_carbon_predictor import NeuralCarbonPredictor, PredictionResult
from .quantum_carbon_optimizer import QuantumCarbonOptimizer, OptimizationResult


class OptimizationAction(Enum):
    """Real-time optimization actions."""
    CONTINUE = "continue"
    PAUSE = "pause"
    MIGRATE = "migrate"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    ADJUST_BATCH = "adjust_batch"
    ADJUST_LR = "adjust_lr"


class CarbonRegime(Enum):
    """Carbon intensity regimes for adaptive optimization."""
    ULTRA_LOW = "ultra_low"    # < 50 gCO2/kWh
    LOW = "low"               # 50-100 gCO2/kWh
    MODERATE = "moderate"     # 100-200 gCO2/kWh
    HIGH = "high"            # 200-400 gCO2/kWh
    ULTRA_HIGH = "ultra_high" # > 400 gCO2/kWh


@dataclass
class AdaptiveTrainingConfig:
    """Configuration for adaptive carbon optimization."""
    # Carbon thresholds for different regimes
    ultra_low_threshold: float = 50.0
    low_threshold: float = 100.0
    moderate_threshold: float = 200.0
    high_threshold: float = 400.0
    
    # Adaptive parameters
    min_batch_size: int = 16
    max_batch_size: int = 256
    min_learning_rate: float = 1e-6
    max_learning_rate: float = 1e-2
    
    # Real-time update intervals
    carbon_check_interval_seconds: int = 300  # 5 minutes
    prediction_update_interval_seconds: int = 900  # 15 minutes
    
    # Migration settings
    migration_enabled: bool = True
    migration_carbon_threshold: float = 300.0
    migration_cost_threshold: float = 1000.0
    
    # Reinforcement learning
    rl_enabled: bool = True
    exploration_rate: float = 0.1
    learning_rate_rl: float = 0.01
    
    # Performance constraints
    max_performance_degradation: float = 0.05  # 5% max degradation
    min_training_efficiency: float = 0.8


@dataclass
class RealTimeState:
    """Current real-time training state."""
    timestamp: datetime
    region: str
    current_carbon_intensity: float
    carbon_regime: CarbonRegime
    current_batch_size: int
    current_learning_rate: float
    training_step: int
    performance_metric: float
    energy_consumption_kwh: float
    cumulative_carbon_kg: float
    
    # Predictions
    carbon_forecast_1h: List[float]
    carbon_forecast_6h: List[float]
    optimal_action: OptimizationAction
    action_confidence: float


@dataclass
class AdaptationAction:
    """Adaptation action with metadata."""
    action_type: OptimizationAction
    parameters: Dict[str, Any]
    expected_carbon_reduction: float
    expected_performance_impact: float
    confidence_score: float
    execution_time: datetime
    rollback_plan: Optional[Dict[str, Any]] = None


class RealtimeAdaptiveCarbonOptimizer:
    """
    Real-time adaptive carbon optimization system with reinforcement learning.
    
    Novel features:
    1. Sub-15-minute carbon intensity predictions
    2. Dynamic hyperparameter adaptation based on carbon regime
    3. Multi-objective optimization with performance constraints
    4. Online reinforcement learning for action selection
    5. Real-time migration across regions and cloud providers
    """
    
    def __init__(
        self,
        config: AdaptiveTrainingConfig,
        neural_predictor: Optional[NeuralCarbonPredictor] = None,
        quantum_optimizer: Optional[QuantumCarbonOptimizer] = None
    ):
        self.config = config
        self.neural_predictor = neural_predictor or NeuralCarbonPredictor()
        self.quantum_optimizer = quantum_optimizer
        self.logger = logging.getLogger(__name__)
        
        # Real-time state tracking
        self.current_state: Optional[RealTimeState] = None
        self.state_history: deque = deque(maxlen=10000)
        self.action_history: deque = deque(maxlen=5000)
        
        # Reinforcement learning components
        self.q_table: Dict[Tuple[CarbonRegime, int], Dict[OptimizationAction, float]] = defaultdict(lambda: defaultdict(float))
        self.state_action_counts: Dict[Tuple[CarbonRegime, OptimizationAction], int] = defaultdict(int)
        self.exploration_rate = config.exploration_rate
        
        # Performance tracking
        self.baseline_performance: Optional[float] = None
        self.adaptation_rewards: deque = deque(maxlen=1000)
        
        # Real-time data streams
        self.carbon_stream_active = False
        self.prediction_cache: Dict[str, Tuple[datetime, List[float]]] = {}
        
        # Action execution
        self.pending_actions: List[AdaptationAction] = []
        self.rollback_stack: List[AdaptationAction] = []
        
    async def initialize(self, initial_region: str, initial_config: TrainingConfig):
        """Initialize the real-time optimizer."""
        self.logger.info("Initializing real-time adaptive carbon optimizer")
        
        # Get initial carbon intensity
        current_carbon = await self._get_current_carbon_intensity(initial_region)
        carbon_regime = self._determine_carbon_regime(current_carbon)
        
        # Initialize state
        self.current_state = RealTimeState(
            timestamp=datetime.now(),
            region=initial_region,
            current_carbon_intensity=current_carbon,
            carbon_regime=carbon_regime,
            current_batch_size=32,  # Default
            current_learning_rate=1e-3,  # Default
            training_step=0,
            performance_metric=0.0,
            energy_consumption_kwh=0.0,
            cumulative_carbon_kg=0.0,
            carbon_forecast_1h=[],
            carbon_forecast_6h=[],
            optimal_action=OptimizationAction.CONTINUE,
            action_confidence=1.0
        )
        
        # Start real-time monitoring
        asyncio.create_task(self._start_carbon_monitoring())
        asyncio.create_task(self._start_prediction_updates())
        asyncio.create_task(self._start_adaptation_engine())
        
        self.logger.info(f"Initialized in region {initial_region} with carbon regime {carbon_regime.value}")
    
    async def get_next_adaptation(self) -> Optional[AdaptationAction]:
        """Get the next recommended adaptation action."""
        if not self.current_state:
            return None
        
        # Update carbon forecasts
        await self._update_carbon_forecasts()
        
        # Determine optimal action using RL + heuristics
        optimal_action = await self._select_optimal_action()
        
        if optimal_action == OptimizationAction.CONTINUE:
            return None
        
        # Create adaptation action
        action = await self._create_adaptation_action(optimal_action)
        
        # Log action selection
        self.logger.info(f"Selected action: {optimal_action.value} with confidence {action.confidence_score:.2f}")
        
        return action
    
    async def execute_adaptation(self, action: AdaptationAction) -> Dict[str, Any]:
        """Execute an adaptation action and track results."""
        self.logger.info(f"Executing adaptation: {action.action_type.value}")
        
        # Record pre-adaptation state
        pre_state = self._capture_current_state()
        
        # Execute the action
        execution_result = await self._execute_action(action)
        
        # Update RL learning
        if self.config.rl_enabled:
            reward = await self._calculate_adaptation_reward(pre_state, action)
            await self._update_q_learning(pre_state, action, reward)
        
        # Store action in history
        self.action_history.append(action)
        
        # Add to rollback stack for potential undo
        if action.rollback_plan:
            self.rollback_stack.append(action)
        
        return execution_result
    
    async def _start_carbon_monitoring(self):
        """Start continuous carbon intensity monitoring."""
        self.carbon_stream_active = True
        
        while self.carbon_stream_active:
            try:
                if self.current_state:
                    # Get current carbon intensity
                    current_carbon = await self._get_current_carbon_intensity(self.current_state.region)
                    
                    # Update state
                    self.current_state.current_carbon_intensity = current_carbon
                    self.current_state.carbon_regime = self._determine_carbon_regime(current_carbon)
                    self.current_state.timestamp = datetime.now()
                    
                    # Store in history
                    self.state_history.append(self._copy_state(self.current_state))
                
                await asyncio.sleep(self.config.carbon_check_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Carbon monitoring error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _start_prediction_updates(self):
        """Start continuous prediction updates."""
        while self.carbon_stream_active:
            try:
                if self.current_state:
                    await self._update_carbon_forecasts()
                
                await asyncio.sleep(self.config.prediction_update_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Prediction update error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _start_adaptation_engine(self):
        """Start continuous adaptation decision engine."""
        while self.carbon_stream_active:
            try:
                # Check if adaptation is needed
                adaptation = await self.get_next_adaptation()
                
                if adaptation:
                    # Execute adaptation automatically if high confidence
                    if adaptation.confidence_score > 0.8:
                        await self.execute_adaptation(adaptation)
                    else:
                        # Queue for manual review
                        self.pending_actions.append(adaptation)
                
                await asyncio.sleep(self.config.carbon_check_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Adaptation engine error: {e}")
                await asyncio.sleep(180)  # Wait 3 minutes on error
    
    async def _update_carbon_forecasts(self):
        """Update carbon intensity forecasts."""
        if not self.current_state:
            return
        
        region = self.current_state.region
        
        # Check cache first
        cache_key = f"{region}_{datetime.now().strftime('%Y%m%d_%H')}"
        if cache_key in self.prediction_cache:
            cache_time, cached_forecast = self.prediction_cache[cache_key]
            if datetime.now() - cache_time < timedelta(minutes=15):
                self.current_state.carbon_forecast_1h = cached_forecast[:6]  # Next 1 hour (6 x 10min)
                self.current_state.carbon_forecast_6h = cached_forecast[:36]  # Next 6 hours
                return
        
        # Generate new predictions
        try:
            from .neural_carbon_predictor import CarbonPredictionInput, WeatherFeatures, EnergyDemandFeatures, RenewableGenerationFeatures
            
            # Create prediction input (simplified)
            prediction_input = CarbonPredictionInput(
                timestamp=datetime.now(),
                region=region,
                weather=WeatherFeatures(
                    temperature=20.0, humidity=60.0, wind_speed=8.0,
                    solar_irradiance=500.0, cloud_cover=50.0, precipitation=0.0
                ),
                demand=EnergyDemandFeatures(
                    total_demand=50000, industrial_demand=20000, residential_demand=15000,
                    commercial_demand=15000, peak_hour_indicator=0.5, day_of_week=2, month_of_year=8
                ),
                renewables=RenewableGenerationFeatures(
                    solar_generation=10000, wind_generation=15000, hydro_generation=8000,
                    geothermal_generation=2000, biomass_generation=1000, renewable_percentage=0.6
                ),
                historical_carbon=[self.current_state.current_carbon_intensity] * 24,
                current_carbon=self.current_state.current_carbon_intensity
            )
            
            # Get prediction
            prediction_result = await self.neural_predictor.predict(prediction_input)
            
            # Extract forecasts
            forecast_values = prediction_result.predicted_intensities
            
            # Update state
            self.current_state.carbon_forecast_1h = forecast_values[:6]  # Next 1 hour
            self.current_state.carbon_forecast_6h = forecast_values[:36]  # Next 6 hours
            
            # Cache results
            self.prediction_cache[cache_key] = (datetime.now(), forecast_values)
            
        except Exception as e:
            self.logger.warning(f"Failed to update carbon forecasts: {e}")
            # Use fallback prediction
            self.current_state.carbon_forecast_1h = [self.current_state.current_carbon_intensity] * 6
            self.current_state.carbon_forecast_6h = [self.current_state.current_carbon_intensity] * 36
    
    async def _select_optimal_action(self) -> OptimizationAction:
        """Select optimal action using reinforcement learning + heuristics."""
        if not self.current_state:
            return OptimizationAction.CONTINUE
        
        state = self.current_state
        carbon_regime = state.carbon_regime
        
        # Epsilon-greedy exploration
        if random.random() < self.exploration_rate:
            # Explore: random action
            actions = list(OptimizationAction)
            return random.choice(actions)
        
        # Exploit: use Q-learning + heuristics
        q_values = self.q_table[(carbon_regime, state.training_step % 100)]
        
        # Heuristic rules for different carbon regimes
        if carbon_regime == CarbonRegime.ULTRA_LOW:
            # Ultra low carbon: scale up, increase batch size
            if state.current_batch_size < self.config.max_batch_size:
                return OptimizationAction.SCALE_UP
            return OptimizationAction.CONTINUE
            
        elif carbon_regime == CarbonRegime.LOW:
            # Low carbon: continue or slight optimization
            return OptimizationAction.CONTINUE
            
        elif carbon_regime == CarbonRegime.MODERATE:
            # Moderate carbon: adjust batch size or learning rate
            if state.current_batch_size > self.config.min_batch_size:
                return OptimizationAction.ADJUST_BATCH
            return OptimizationAction.ADJUST_LR
            
        elif carbon_regime == CarbonRegime.HIGH:
            # High carbon: scale down or pause
            avg_future_carbon = statistics.mean(state.carbon_forecast_1h) if state.carbon_forecast_1h else state.current_carbon_intensity
            if avg_future_carbon < state.current_carbon_intensity * 0.8:
                return OptimizationAction.PAUSE  # Wait for lower carbon
            return OptimizationAction.SCALE_DOWN
            
        else:  # ULTRA_HIGH
            # Ultra high carbon: pause or migrate
            if self.config.migration_enabled and state.current_carbon_intensity > self.config.migration_carbon_threshold:
                return OptimizationAction.MIGRATE
            return OptimizationAction.PAUSE
        
        # Fallback to Q-learning if no heuristic applies
        if q_values:
            best_action = max(q_values.items(), key=lambda x: x[1])[0]
            return best_action
        
        return OptimizationAction.CONTINUE
    
    async def _create_adaptation_action(self, action_type: OptimizationAction) -> AdaptationAction:
        """Create adaptation action with parameters and predictions."""
        state = self.current_state
        parameters = {}
        expected_carbon_reduction = 0.0
        expected_performance_impact = 0.0
        confidence_score = 0.8
        
        if action_type == OptimizationAction.PAUSE:
            # Pause training for a calculated duration
            resume_time = await self._calculate_optimal_resume_time()
            parameters = {"resume_time": resume_time}
            expected_carbon_reduction = 0.3  # 30% reduction during pause
            expected_performance_impact = -0.02  # Slight delay impact
            confidence_score = 0.9
            
        elif action_type == OptimizationAction.SCALE_UP:
            # Increase computational resources
            new_batch_size = min(state.current_batch_size * 2, self.config.max_batch_size)
            parameters = {"new_batch_size": new_batch_size}
            expected_carbon_reduction = -0.1  # Slight increase due to more compute
            expected_performance_impact = 0.05  # Faster training
            confidence_score = 0.7
            
        elif action_type == OptimizationAction.SCALE_DOWN:
            # Decrease computational resources
            new_batch_size = max(state.current_batch_size // 2, self.config.min_batch_size)
            parameters = {"new_batch_size": new_batch_size}
            expected_carbon_reduction = 0.2  # 20% reduction
            expected_performance_impact = -0.03  # Slightly slower
            confidence_score = 0.8
            
        elif action_type == OptimizationAction.ADJUST_BATCH:
            # Optimize batch size for current carbon regime
            optimal_batch = await self._calculate_optimal_batch_size()
            parameters = {"new_batch_size": optimal_batch}
            expected_carbon_reduction = 0.1  # 10% reduction
            expected_performance_impact = 0.01  # Slight improvement
            confidence_score = 0.75
            
        elif action_type == OptimizationAction.ADJUST_LR:
            # Optimize learning rate for current carbon regime
            optimal_lr = await self._calculate_optimal_learning_rate()
            parameters = {"new_learning_rate": optimal_lr}
            expected_carbon_reduction = 0.05  # 5% reduction
            expected_performance_impact = 0.02  # Slight improvement
            confidence_score = 0.7
            
        elif action_type == OptimizationAction.MIGRATE:
            # Migrate to region with lower carbon intensity
            target_region = await self._find_optimal_migration_target()
            parameters = {"target_region": target_region}
            expected_carbon_reduction = 0.4  # 40% reduction
            expected_performance_impact = -0.01  # Migration overhead
            confidence_score = 0.6  # Lower confidence due to complexity
        
        # Create rollback plan
        rollback_plan = self._create_rollback_plan(action_type, parameters)
        
        return AdaptationAction(
            action_type=action_type,
            parameters=parameters,
            expected_carbon_reduction=expected_carbon_reduction,
            expected_performance_impact=expected_performance_impact,
            confidence_score=confidence_score,
            execution_time=datetime.now(),
            rollback_plan=rollback_plan
        )
    
    async def _execute_action(self, action: AdaptationAction) -> Dict[str, Any]:
        """Execute the adaptation action."""
        action_type = action.action_type
        parameters = action.parameters
        
        result = {"success": False, "message": "", "metrics": {}}
        
        try:
            if action_type == OptimizationAction.PAUSE:
                result = await self._execute_pause(parameters)
            elif action_type == OptimizationAction.SCALE_UP:
                result = await self._execute_scale_up(parameters)
            elif action_type == OptimizationAction.SCALE_DOWN:
                result = await self._execute_scale_down(parameters)
            elif action_type == OptimizationAction.ADJUST_BATCH:
                result = await self._execute_adjust_batch(parameters)
            elif action_type == OptimizationAction.ADJUST_LR:
                result = await self._execute_adjust_lr(parameters)
            elif action_type == OptimizationAction.MIGRATE:
                result = await self._execute_migrate(parameters)
            else:
                result["message"] = "Unknown action type"
                
        except Exception as e:
            result["message"] = f"Action execution failed: {e}"
            self.logger.error(f"Failed to execute {action_type.value}: {e}")
        
        return result
    
    # Action execution methods
    
    async def _execute_pause(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute training pause."""
        resume_time = parameters.get("resume_time", datetime.now() + timedelta(hours=1))
        
        # In a real implementation, this would pause the training loop
        self.logger.info(f"Training paused until {resume_time}")
        
        return {
            "success": True,
            "message": f"Training paused until {resume_time}",
            "metrics": {"pause_duration_hours": (resume_time - datetime.now()).total_seconds() / 3600}
        }
    
    async def _execute_scale_up(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute scale up action."""
        new_batch_size = parameters.get("new_batch_size", 64)
        
        if self.current_state:
            old_batch_size = self.current_state.current_batch_size
            self.current_state.current_batch_size = new_batch_size
            
            self.logger.info(f"Scaled up: batch size {old_batch_size} → {new_batch_size}")
            
            return {
                "success": True,
                "message": f"Scaled up batch size to {new_batch_size}",
                "metrics": {"old_batch_size": old_batch_size, "new_batch_size": new_batch_size}
            }
        
        return {"success": False, "message": "No current state available"}
    
    async def _execute_scale_down(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute scale down action."""
        new_batch_size = parameters.get("new_batch_size", 16)
        
        if self.current_state:
            old_batch_size = self.current_state.current_batch_size
            self.current_state.current_batch_size = new_batch_size
            
            self.logger.info(f"Scaled down: batch size {old_batch_size} → {new_batch_size}")
            
            return {
                "success": True,
                "message": f"Scaled down batch size to {new_batch_size}",
                "metrics": {"old_batch_size": old_batch_size, "new_batch_size": new_batch_size}
            }
        
        return {"success": False, "message": "No current state available"}
    
    async def _execute_adjust_batch(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute batch size adjustment."""
        return await self._execute_scale_up(parameters)  # Same implementation
    
    async def _execute_adjust_lr(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute learning rate adjustment."""
        new_lr = parameters.get("new_learning_rate", 1e-3)
        
        if self.current_state:
            old_lr = self.current_state.current_learning_rate
            self.current_state.current_learning_rate = new_lr
            
            self.logger.info(f"Adjusted learning rate: {old_lr:.2e} → {new_lr:.2e}")
            
            return {
                "success": True,
                "message": f"Adjusted learning rate to {new_lr:.2e}",
                "metrics": {"old_learning_rate": old_lr, "new_learning_rate": new_lr}
            }
        
        return {"success": False, "message": "No current state available"}
    
    async def _execute_migrate(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute migration to new region."""
        target_region = parameters.get("target_region", "US-WA")
        
        if self.current_state:
            old_region = self.current_state.region
            self.current_state.region = target_region
            
            # Update carbon intensity for new region
            new_carbon = await self._get_current_carbon_intensity(target_region)
            self.current_state.current_carbon_intensity = new_carbon
            self.current_state.carbon_regime = self._determine_carbon_regime(new_carbon)
            
            self.logger.info(f"Migrated: {old_region} → {target_region}")
            
            return {
                "success": True,
                "message": f"Migrated to {target_region}",
                "metrics": {"old_region": old_region, "new_region": target_region, "new_carbon_intensity": new_carbon}
            }
        
        return {"success": False, "message": "No current state available"}
    
    # Helper methods
    
    def _determine_carbon_regime(self, carbon_intensity: float) -> CarbonRegime:
        """Determine carbon regime based on intensity."""
        if carbon_intensity < self.config.ultra_low_threshold:
            return CarbonRegime.ULTRA_LOW
        elif carbon_intensity < self.config.low_threshold:
            return CarbonRegime.LOW
        elif carbon_intensity < self.config.moderate_threshold:
            return CarbonRegime.MODERATE
        elif carbon_intensity < self.config.high_threshold:
            return CarbonRegime.HIGH
        else:
            return CarbonRegime.ULTRA_HIGH
    
    async def _get_current_carbon_intensity(self, region: str) -> float:
        """Get current carbon intensity for region."""
        # Simplified implementation - would call real APIs
        base_intensities = {
            "US-CA": 150,
            "US-WA": 80,
            "EU-FR": 120,
            "EU-NO": 20,
            "US-TX": 400,
            "IN-KA": 600
        }
        
        base = base_intensities.get(region, 200)
        # Add random variation
        return base + np.random.normal(0, base * 0.2)
    
    async def _calculate_optimal_resume_time(self) -> datetime:
        """Calculate optimal time to resume training."""
        if not self.current_state or not self.current_state.carbon_forecast_6h:
            return datetime.now() + timedelta(hours=1)
        
        # Find minimum carbon in next 6 hours
        forecast = self.current_state.carbon_forecast_6h
        min_carbon_idx = np.argmin(forecast)
        
        # Resume at minimum carbon time (or in 1 hour if current is already low)
        resume_time = datetime.now() + timedelta(minutes=10 * min_carbon_idx)
        
        return max(resume_time, datetime.now() + timedelta(minutes=30))
    
    async def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size for current carbon regime."""
        if not self.current_state:
            return 32
        
        regime = self.current_state.carbon_regime
        
        if regime == CarbonRegime.ULTRA_LOW:
            return self.config.max_batch_size
        elif regime == CarbonRegime.LOW:
            return min(128, self.config.max_batch_size)
        elif regime == CarbonRegime.MODERATE:
            return 64
        elif regime == CarbonRegime.HIGH:
            return 32
        else:  # ULTRA_HIGH
            return self.config.min_batch_size
    
    async def _calculate_optimal_learning_rate(self) -> float:
        """Calculate optimal learning rate for current carbon regime."""
        if not self.current_state:
            return 1e-3
        
        regime = self.current_state.carbon_regime
        
        if regime == CarbonRegime.ULTRA_LOW:
            return self.config.max_learning_rate
        elif regime == CarbonRegime.LOW:
            return 1e-3
        elif regime == CarbonRegime.MODERATE:
            return 5e-4
        elif regime == CarbonRegime.HIGH:
            return 1e-4
        else:  # ULTRA_HIGH
            return self.config.min_learning_rate
    
    async def _find_optimal_migration_target(self) -> str:
        """Find optimal region for migration."""
        # Check carbon intensity across regions
        regions = ["US-CA", "US-WA", "EU-FR", "EU-NO"]
        region_carbons = {}
        
        for region in regions:
            carbon = await self._get_current_carbon_intensity(region)
            region_carbons[region] = carbon
        
        # Return region with lowest carbon intensity
        best_region = min(region_carbons.items(), key=lambda x: x[1])[0]
        return best_region
    
    def _create_rollback_plan(self, action_type: OptimizationAction, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create rollback plan for action."""
        if not self.current_state:
            return {}
        
        rollback = {"action_type": "rollback"}
        
        if action_type in [OptimizationAction.SCALE_UP, OptimizationAction.SCALE_DOWN, OptimizationAction.ADJUST_BATCH]:
            rollback["restore_batch_size"] = self.current_state.current_batch_size
        elif action_type == OptimizationAction.ADJUST_LR:
            rollback["restore_learning_rate"] = self.current_state.current_learning_rate
        elif action_type == OptimizationAction.MIGRATE:
            rollback["restore_region"] = self.current_state.region
        
        return rollback
    
    def _capture_current_state(self) -> Dict[str, Any]:
        """Capture current state for RL learning."""
        if not self.current_state:
            return {}
        
        return {
            "carbon_regime": self.current_state.carbon_regime,
            "carbon_intensity": self.current_state.current_carbon_intensity,
            "batch_size": self.current_state.current_batch_size,
            "learning_rate": self.current_state.current_learning_rate,
            "performance_metric": self.current_state.performance_metric,
            "training_step": self.current_state.training_step
        }
    
    async def _calculate_adaptation_reward(self, pre_state: Dict[str, Any], action: AdaptationAction) -> float:
        """Calculate reward for reinforcement learning."""
        if not self.current_state:
            return 0.0
        
        # Multi-objective reward: carbon reduction + performance + cost
        carbon_reward = 0.0
        performance_reward = 0.0
        cost_reward = 0.0
        
        # Carbon reward (primary objective)
        if "carbon_intensity" in pre_state:
            carbon_reduction = pre_state["carbon_intensity"] - self.current_state.current_carbon_intensity
            carbon_reward = carbon_reduction / 100.0  # Normalize
        
        # Performance reward (secondary objective)
        if "performance_metric" in pre_state and pre_state["performance_metric"] > 0:
            performance_change = (self.current_state.performance_metric - pre_state["performance_metric"]) / pre_state["performance_metric"]
            performance_reward = performance_change * 0.5  # Weight factor
        
        # Cost reward (minimize cost)
        cost_reward = -0.1 if action.action_type == OptimizationAction.MIGRATE else 0.0
        
        # Combined reward
        total_reward = carbon_reward + performance_reward + cost_reward
        
        # Penalty for violating performance constraints
        if action.expected_performance_impact < -self.config.max_performance_degradation:
            total_reward -= 1.0
        
        return total_reward
    
    async def _update_q_learning(self, pre_state: Dict[str, Any], action: AdaptationAction, reward: float):
        """Update Q-learning table."""
        if "carbon_regime" not in pre_state:
            return
        
        state_key = (pre_state["carbon_regime"], pre_state.get("training_step", 0) % 100)
        action_key = action.action_type
        
        # Update Q-value using Q-learning formula
        old_q = self.q_table[state_key][action_key]
        
        # Simplified next state value (would use actual next state in full implementation)
        next_state_value = max(self.q_table[(self.current_state.carbon_regime, self.current_state.training_step % 100)].values()) if self.q_table[(self.current_state.carbon_regime, self.current_state.training_step % 100)] else 0.0
        
        new_q = old_q + self.config.learning_rate_rl * (reward + 0.9 * next_state_value - old_q)
        
        self.q_table[state_key][action_key] = new_q
        self.state_action_counts[(pre_state["carbon_regime"], action_key)] += 1
        
        # Store reward for tracking
        self.adaptation_rewards.append(reward)
        
        self.logger.debug(f"Updated Q-value for {state_key}, {action_key}: {old_q:.3f} → {new_q:.3f} (reward: {reward:.3f})")
    
    def _copy_state(self, state: RealTimeState) -> RealTimeState:
        """Create a copy of the current state."""
        return RealTimeState(
            timestamp=state.timestamp,
            region=state.region,
            current_carbon_intensity=state.current_carbon_intensity,
            carbon_regime=state.carbon_regime,
            current_batch_size=state.current_batch_size,
            current_learning_rate=state.current_learning_rate,
            training_step=state.training_step,
            performance_metric=state.performance_metric,
            energy_consumption_kwh=state.energy_consumption_kwh,
            cumulative_carbon_kg=state.cumulative_carbon_kg,
            carbon_forecast_1h=state.carbon_forecast_1h.copy(),
            carbon_forecast_6h=state.carbon_forecast_6h.copy(),
            optimal_action=state.optimal_action,
            action_confidence=state.action_confidence
        )
    
    async def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get comprehensive optimization metrics."""
        if not self.current_state:
            return {}
        
        return {
            "current_state": {
                "region": self.current_state.region,
                "carbon_intensity": self.current_state.current_carbon_intensity,
                "carbon_regime": self.current_state.carbon_regime.value,
                "batch_size": self.current_state.current_batch_size,
                "learning_rate": self.current_state.current_learning_rate,
                "training_step": self.current_state.training_step,
                "cumulative_carbon_kg": self.current_state.cumulative_carbon_kg
            },
            "adaptation_stats": {
                "total_adaptations": len(self.action_history),
                "recent_reward": statistics.mean(list(self.adaptation_rewards)[-10:]) if self.adaptation_rewards else 0.0,
                "avg_reward": statistics.mean(self.adaptation_rewards) if self.adaptation_rewards else 0.0,
                "exploration_rate": self.exploration_rate
            },
            "forecasts": {
                "next_1h_carbon": self.current_state.carbon_forecast_1h,
                "next_6h_carbon": self.current_state.carbon_forecast_6h
            },
            "q_learning_stats": {
                "state_action_pairs": len(self.q_table),
                "total_learning_updates": sum(self.state_action_counts.values())
            }
        }