"""
Real-time carbon optimization engine for dynamic training adjustments.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

import numpy as np
from pydantic import BaseModel

from .types import CarbonIntensity, CarbonForecast, TrainingMetrics
from .monitor import CarbonMonitor
from .exceptions import CarbonAwareException


logger = logging.getLogger(__name__)


class OptimizationMode(Enum):
    """Optimization modes for real-time adjustments."""
    CARBON_FIRST = "carbon_first"        # Minimize carbon at all costs
    BALANCED = "balanced"                # Balance carbon, cost, and performance
    PERFORMANCE_FIRST = "performance_first"  # Minimize time, consider carbon
    COST_AWARE = "cost_aware"           # Balance carbon and cost


class TrainingState(Enum):
    """Current state of training."""
    RUNNING = "running"
    PAUSED = "paused"
    SCALING_UP = "scaling_up"
    SCALING_DOWN = "scaling_down"
    MIGRATING = "migrating"


@dataclass
class OptimizationParameters:
    """Parameters for real-time optimization."""
    carbon_weight: float = 1.0           # Weight for carbon in optimization
    cost_weight: float = 0.3             # Weight for cost considerations
    performance_weight: float = 0.5      # Weight for performance/time
    max_pause_minutes: int = 60          # Maximum pause duration
    min_batch_size: int = 8              # Minimum batch size
    max_batch_size: int = 512            # Maximum batch size
    gpu_scaling_factor: float = 2.0      # Factor for GPU scaling decisions


class OptimizationAction(BaseModel):
    """Action recommended by the optimizer."""
    action_type: str
    parameters: Dict[str, Any]
    confidence: float  # 0-1 confidence in this action
    expected_carbon_savings: float
    expected_cost_impact: float
    expected_time_impact: float
    rationale: str


class RealTimeOptimizer:
    """Real-time optimizer for carbon-aware training adjustments."""
    
    def __init__(
        self,
        monitor: Optional[CarbonMonitor] = None,
        mode: OptimizationMode = OptimizationMode.BALANCED,
        optimization_interval: int = 300  # seconds
    ):
        self.monitor = monitor or CarbonMonitor()
        self.mode = mode
        self.optimization_interval = optimization_interval
        self.current_state = TrainingState.RUNNING
        self.optimization_params = OptimizationParameters()
        
        # Tracking
        self.optimization_history = []
        self.carbon_history = []
        self.performance_metrics = {}
        self._running = False
        self._optimization_task = None
        
    async def start_optimization(
        self,
        region: str,
        training_callback: Optional[Callable] = None
    ) -> None:
        """Start real-time optimization loop."""
        if self._running:
            logger.warning("Optimization already running")
            return
            
        self._running = True
        self.region = region
        self.training_callback = training_callback
        
        logger.info(f"Starting real-time optimization for {region} in {self.mode.value} mode")
        
        self._optimization_task = asyncio.create_task(self._optimization_loop())
        
    async def stop_optimization(self) -> None:
        """Stop the optimization loop."""
        self._running = False
        if self._optimization_task:
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Optimization stopped")
        
    async def _optimization_loop(self) -> None:
        """Main optimization loop."""
        while self._running:
            try:
                # Get current carbon data
                current_carbon = await self.monitor.get_current_intensity(self.region)
                if current_carbon:
                    self.carbon_history.append({
                        'timestamp': datetime.now(),
                        'carbon_intensity': current_carbon.carbon_intensity,
                        'energy_mix': current_carbon.energy_mix
                    })
                
                # Get forecast for next few hours
                forecast = await self.monitor.get_forecast(self.region, hours=4)
                
                # Analyze and decide on optimization action
                action = await self._analyze_and_optimize(current_carbon, forecast)
                
                if action:
                    await self._execute_action(action)
                    self.optimization_history.append({
                        'timestamp': datetime.now(),
                        'action': action.dict(),
                        'carbon_intensity': current_carbon.carbon_intensity if current_carbon else None
                    })
                    
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                
            # Wait for next optimization cycle
            await asyncio.sleep(self.optimization_interval)
            
    async def _analyze_and_optimize(
        self,
        current_carbon: Optional[CarbonIntensity],
        forecast: Optional[CarbonForecast]
    ) -> Optional[OptimizationAction]:
        """Analyze current conditions and determine optimal action."""
        if not current_carbon:
            return None
            
        carbon_intensity = current_carbon.carbon_intensity
        
        # Analyze trend
        carbon_trend = self._analyze_carbon_trend()
        forecast_trend = self._analyze_forecast_trend(forecast) if forecast else 0
        
        # Decision logic based on mode
        if self.mode == OptimizationMode.CARBON_FIRST:
            return await self._carbon_first_optimization(carbon_intensity, carbon_trend, forecast_trend)
        elif self.mode == OptimizationMode.BALANCED:
            return await self._balanced_optimization(carbon_intensity, carbon_trend, forecast_trend)
        elif self.mode == OptimizationMode.PERFORMANCE_FIRST:
            return await self._performance_first_optimization(carbon_intensity, carbon_trend, forecast_trend)
        elif self.mode == OptimizationMode.COST_AWARE:
            return await self._cost_aware_optimization(carbon_intensity, carbon_trend, forecast_trend)
            
        return None
        
    def _analyze_carbon_trend(self) -> float:
        """Analyze recent carbon intensity trend."""
        if len(self.carbon_history) < 2:
            return 0.0
            
        recent_data = self.carbon_history[-10:]  # Last 10 measurements
        intensities = [d['carbon_intensity'] for d in recent_data]
        
        # Simple linear trend calculation
        x = np.arange(len(intensities))
        slope = np.polyfit(x, intensities, 1)[0] if len(intensities) > 1 else 0
        
        return slope
        
    def _analyze_forecast_trend(self, forecast: CarbonForecast) -> float:
        """Analyze forecast trend."""
        if not forecast or len(forecast.data_points) < 2:
            return 0.0
            
        # Look at next 2 hours
        near_term = forecast.data_points[:2]
        intensities = [dp.carbon_intensity for dp in near_term]
        
        if len(intensities) >= 2:
            return intensities[1] - intensities[0]
            
        return 0.0
        
    async def _carbon_first_optimization(
        self,
        carbon_intensity: float,
        carbon_trend: float,
        forecast_trend: float
    ) -> Optional[OptimizationAction]:
        """Optimization prioritizing carbon reduction."""
        
        # High carbon intensity - pause training
        if carbon_intensity > 150:
            return OptimizationAction(
                action_type="pause_training",
                parameters={"duration_minutes": min(30, self.optimization_params.max_pause_minutes)},
                confidence=0.9,
                expected_carbon_savings=carbon_intensity * 0.5,  # Rough estimate
                expected_cost_impact=0.0,
                expected_time_impact=0.5,  # 50% slower due to pause
                rationale=f"High carbon intensity ({carbon_intensity:.1f} gCO2/kWh) - pausing to wait for cleaner energy"
            )
            
        # Carbon rising - reduce batch size
        elif carbon_trend > 10 or forecast_trend > 20:
            return OptimizationAction(
                action_type="adjust_batch_size",
                parameters={"factor": 0.7},  # Reduce by 30%
                confidence=0.7,
                expected_carbon_savings=carbon_intensity * 0.2,
                expected_cost_impact=0.1,
                expected_time_impact=0.2,
                rationale="Carbon intensity rising - reducing computational load"
            )
            
        # Carbon decreasing - can increase utilization
        elif carbon_trend < -10 and carbon_intensity < 100:
            return OptimizationAction(
                action_type="adjust_batch_size", 
                parameters={"factor": 1.3},  # Increase by 30%
                confidence=0.6,
                expected_carbon_savings=-carbon_intensity * 0.1,  # Negative = increase
                expected_cost_impact=-0.1,
                expected_time_impact=-0.2,  # Negative = faster
                rationale="Carbon intensity low and decreasing - increasing utilization"
            )
            
        return None
        
    async def _balanced_optimization(
        self,
        carbon_intensity: float,
        carbon_trend: float,
        forecast_trend: float
    ) -> Optional[OptimizationAction]:
        """Balanced optimization considering carbon, cost, and performance."""
        
        # Extreme carbon - pause regardless of cost
        if carbon_intensity > 200:
            return OptimizationAction(
                action_type="pause_training",
                parameters={"duration_minutes": 20},
                confidence=0.8,
                expected_carbon_savings=carbon_intensity * 0.6,
                expected_cost_impact=0.0,
                expected_time_impact=0.3,
                rationale=f"Very high carbon intensity ({carbon_intensity:.1f} gCO2/kWh) - temporary pause"
            )
            
        # Moderately high carbon - reduce batch size
        elif carbon_intensity > 120:
            return OptimizationAction(
                action_type="adjust_batch_size",
                parameters={"factor": 0.8},
                confidence=0.6,
                expected_carbon_savings=carbon_intensity * 0.15,
                expected_cost_impact=0.05,
                expected_time_impact=0.15,
                rationale="Moderately high carbon - slight reduction in compute"
            )
            
        # Low carbon with good forecast - optimize for performance
        elif carbon_intensity < 80 and forecast_trend < 10:
            return OptimizationAction(
                action_type="adjust_batch_size",
                parameters={"factor": 1.2},
                confidence=0.7,
                expected_carbon_savings=-carbon_intensity * 0.1,
                expected_cost_impact=-0.05,
                expected_time_impact=-0.15,
                rationale="Low carbon intensity with stable forecast - optimizing performance"
            )
            
        return None
        
    async def _performance_first_optimization(
        self,
        carbon_intensity: float,
        carbon_trend: float,
        forecast_trend: float
    ) -> Optional[OptimizationAction]:
        """Performance-first optimization with carbon consideration."""
        
        # Only pause for extremely high carbon
        if carbon_intensity > 300:
            return OptimizationAction(
                action_type="pause_training",
                parameters={"duration_minutes": 10},
                confidence=0.6,
                expected_carbon_savings=carbon_intensity * 0.5,
                expected_cost_impact=0.0,
                expected_time_impact=0.2,
                rationale=f"Extremely high carbon ({carbon_intensity:.1f} gCO2/kWh) - brief pause"
            )
            
        # Focus on performance optimization
        elif carbon_intensity < 150:  # Acceptable carbon level
            return OptimizationAction(
                action_type="adjust_batch_size",
                parameters={"factor": 1.25},
                confidence=0.8,
                expected_carbon_savings=-carbon_intensity * 0.15,
                expected_cost_impact=-0.1,
                expected_time_impact=-0.2,
                rationale="Carbon acceptable - optimizing for performance"
            )
            
        return None
        
    async def _cost_aware_optimization(
        self,
        carbon_intensity: float,
        carbon_trend: float,
        forecast_trend: float
    ) -> Optional[OptimizationAction]:
        """Cost-aware optimization balancing carbon and economics."""
        
        # High carbon with rising trend - pause to save money and carbon
        if carbon_intensity > 180 and (carbon_trend > 15 or forecast_trend > 25):
            return OptimizationAction(
                action_type="pause_training",
                parameters={"duration_minutes": 25},
                confidence=0.8,
                expected_carbon_savings=carbon_intensity * 0.5,
                expected_cost_impact=0.2,  # Positive = savings
                expected_time_impact=0.4,
                rationale="High carbon with rising trend - pause for cost and carbon savings"
            )
            
        # Moderate carbon - slight adjustment
        elif carbon_intensity > 100:
            return OptimizationAction(
                action_type="adjust_batch_size",
                parameters={"factor": 0.85},
                confidence=0.6,
                expected_carbon_savings=carbon_intensity * 0.1,
                expected_cost_impact=0.05,
                expected_time_impact=0.1,
                rationale="Moderate carbon - slight efficiency adjustment"
            )
            
        return None
        
    async def _execute_action(self, action: OptimizationAction) -> None:
        """Execute the recommended optimization action."""
        logger.info(f"Executing action: {action.action_type} - {action.rationale}")
        
        try:
            if action.action_type == "pause_training":
                await self._pause_training(action.parameters.get("duration_minutes", 10))
                
            elif action.action_type == "adjust_batch_size":
                await self._adjust_batch_size(action.parameters.get("factor", 1.0))
                
            elif action.action_type == "scale_gpus":
                await self._scale_gpus(action.parameters.get("factor", 1.0))
                
            # Execute callback if provided
            if self.training_callback:
                await self.training_callback(action)
                
        except Exception as e:
            logger.error(f"Failed to execute action {action.action_type}: {e}")
            
    async def _pause_training(self, duration_minutes: int) -> None:
        """Pause training for specified duration."""
        logger.info(f"Pausing training for {duration_minutes} minutes")
        self.current_state = TrainingState.PAUSED
        
        # In real implementation, this would pause actual training
        await asyncio.sleep(min(duration_minutes * 60, 300))  # Cap at 5 minutes for demo
        
        self.current_state = TrainingState.RUNNING
        logger.info("Resuming training after carbon-aware pause")
        
    async def _adjust_batch_size(self, factor: float) -> None:
        """Adjust training batch size."""
        logger.info(f"Adjusting batch size by factor {factor:.2f}")
        
        # In real implementation, this would adjust actual batch size
        # For now, just simulate the adjustment
        await asyncio.sleep(1)
        
    async def _scale_gpus(self, factor: float) -> None:
        """Scale GPU allocation."""
        if factor > 1.0:
            self.current_state = TrainingState.SCALING_UP
            logger.info(f"Scaling up GPUs by factor {factor:.2f}")
        else:
            self.current_state = TrainingState.SCALING_DOWN
            logger.info(f"Scaling down GPUs by factor {factor:.2f}")
            
        # Simulate scaling operation
        await asyncio.sleep(2)
        self.current_state = TrainingState.RUNNING
        
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization activities."""
        if not self.optimization_history:
            return {"message": "No optimizations performed yet"}
            
        total_actions = len(self.optimization_history)
        action_types = {}
        total_carbon_savings = 0.0
        
        for opt in self.optimization_history:
            action_type = opt['action']['action_type']
            action_types[action_type] = action_types.get(action_type, 0) + 1
            total_carbon_savings += opt['action'].get('expected_carbon_savings', 0)
            
        return {
            "total_optimizations": total_actions,
            "action_breakdown": action_types,
            "estimated_carbon_savings_kg": total_carbon_savings / 1000,  # Convert to kg
            "current_state": self.current_state.value,
            "optimization_mode": self.mode.value,
            "carbon_measurements": len(self.carbon_history)
        }
        
    def get_carbon_timeline(self) -> List[Dict[str, Any]]:
        """Get carbon intensity timeline."""
        return self.carbon_history.copy()
        
    def update_optimization_params(self, **kwargs) -> None:
        """Update optimization parameters."""
        for key, value in kwargs.items():
            if hasattr(self.optimization_params, key):
                setattr(self.optimization_params, key, value)
                logger.info(f"Updated {key} to {value}")