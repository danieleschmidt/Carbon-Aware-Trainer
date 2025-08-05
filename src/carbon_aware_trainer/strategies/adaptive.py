"""Adaptive carbon-aware scheduling strategy with machine learning."""

import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import logging
import numpy as np
from pathlib import Path

from ..core.types import TrainingState, CarbonIntensity, OptimalWindow
from ..core.monitor import CarbonMonitor


logger = logging.getLogger(__name__)


class AdaptiveScheduler:
    """Adaptive scheduler that learns from historical patterns."""
    
    def __init__(
        self,
        historical_data_path: Optional[str] = None,
        workload_flexibility: float = 0.3,
        prediction_model: str = 'linear',
        learning_rate: float = 0.01,
        adaptation_window_hours: int = 168  # 1 week
    ):
        """Initialize adaptive scheduler.
        
        Args:
            historical_data_path: Path to historical carbon/training data
            workload_flexibility: Flexibility in scheduling (0-1)
            prediction_model: Prediction model type ('linear', 'moving_average')
            learning_rate: Learning rate for adaptation
            adaptation_window_hours: Window for pattern learning
        """
        self.workload_flexibility = workload_flexibility
        self.prediction_model = prediction_model
        self.learning_rate = learning_rate
        self.adaptation_window_hours = adaptation_window_hours
        
        # Historical data and patterns
        self.historical_data: List[Dict[str, Any]] = []
        self.daily_patterns: Dict[int, List[float]] = {}  # Hour -> avg intensities
        self.weekly_patterns: Dict[int, List[float]] = {}  # Weekday -> avg intensities
        
        # Adaptive thresholds
        self.base_threshold = 100.0
        self.adaptive_threshold = 100.0
        self.threshold_history: List[Tuple[datetime, float]] = []
        
        # Load historical data if provided
        if historical_data_path:
            self._load_historical_data(historical_data_path)
            self._learn_patterns()
    
    def _load_historical_data(self, data_path: str) -> None:
        """Load historical carbon and training data."""
        try:
            path = Path(data_path)
            if path.exists():
                if path.suffix.lower() == '.json':
                    with open(path, 'r') as f:
                        self.historical_data = json.load(f)
                elif path.suffix.lower() == '.csv':
                    import pandas as pd
                    df = pd.read_csv(path)
                    self.historical_data = df.to_dict('records')
                
                logger.info(f"Loaded {len(self.historical_data)} historical data points")
            else:
                logger.warning(f"Historical data file not found: {data_path}")
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
    
    def _learn_patterns(self) -> None:
        """Learn daily and weekly patterns from historical data."""
        if not self.historical_data:
            return
        
        # Group data by hour and weekday
        hourly_data: Dict[int, List[float]] = {i: [] for i in range(24)}
        weekly_data: Dict[int, List[float]] = {i: [] for i in range(7)}
        
        for record in self.historical_data:
            try:
                timestamp = datetime.fromisoformat(record['timestamp'])
                intensity = float(record['carbon_intensity'])
                
                hourly_data[timestamp.hour].append(intensity)
                weekly_data[timestamp.weekday()].append(intensity)
            except (KeyError, ValueError) as e:
                logger.debug(f"Skipping invalid record: {e}")
        
        # Calculate patterns
        self.daily_patterns = {
            hour: intensities for hour, intensities in hourly_data.items()
            if intensities
        }
        self.weekly_patterns = {
            day: intensities for day, intensities in weekly_data.items()
            if intensities
        }
        
        logger.info(f"Learned patterns for {len(self.daily_patterns)} hours and {len(self.weekly_patterns)} weekdays")
    
    def predict_carbon_intensity(
        self, 
        region: str, 
        target_time: datetime,
        current_intensity: Optional[CarbonIntensity] = None
    ) -> float:
        """Predict carbon intensity for a future time.
        
        Args:
            region: Region code
            target_time: Time to predict for
            current_intensity: Current carbon intensity for reference
            
        Returns:
            Predicted carbon intensity
        """
        if self.prediction_model == 'moving_average':
            return self._predict_moving_average(target_time, current_intensity)
        elif self.prediction_model == 'linear':
            return self._predict_linear_trend(target_time, current_intensity)
        else:
            # Fallback to pattern-based prediction
            return self._predict_pattern_based(target_time, current_intensity)
    
    def _predict_moving_average(
        self, 
        target_time: datetime,
        current_intensity: Optional[CarbonIntensity]
    ) -> float:
        """Predict using moving average of similar times."""
        hour = target_time.hour
        weekday = target_time.weekday()
        
        # Use daily pattern if available
        if hour in self.daily_patterns:
            daily_avg = np.mean(self.daily_patterns[hour])
        else:
            daily_avg = 120.0  # Default assumption
        
        # Use weekly pattern if available
        if weekday in self.weekly_patterns:
            weekly_avg = np.mean(self.weekly_patterns[weekday])
        else:
            weekly_avg = daily_avg
        
        # Combine patterns
        predicted = (daily_avg * 0.7) + (weekly_avg * 0.3)
        
        # Adjust based on current intensity if available
        if current_intensity:
            # Simple trend continuation
            trend_factor = 0.1
            predicted = predicted * (1 - trend_factor) + current_intensity.carbon_intensity * trend_factor
        
        return max(10.0, predicted)  # Minimum reasonable value
    
    def _predict_linear_trend(
        self, 
        target_time: datetime,
        current_intensity: Optional[CarbonIntensity]
    ) -> float:
        """Predict using linear trend analysis."""
        if not current_intensity:
            return self._predict_pattern_based(target_time, current_intensity)
        
        # Simple linear trend from recent threshold adjustments
        if len(self.threshold_history) >= 2:
            recent_thresholds = self.threshold_history[-5:]  # Last 5 adjustments
            
            if len(recent_thresholds) >= 2:
                # Calculate trend
                times = [(t[0] - recent_thresholds[0][0]).total_seconds() for t in recent_thresholds]
                values = [t[1] for t in recent_thresholds]
                
                # Simple linear regression
                n = len(times)
                sum_x = sum(times)
                sum_y = sum(values)
                sum_xy = sum(x * y for x, y in zip(times, values))
                sum_x2 = sum(x * x for x in times)
                
                if n * sum_x2 - sum_x * sum_x != 0:
                    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                    intercept = (sum_y - slope * sum_x) / n
                    
                    # Project to target time
                    time_diff = (target_time - recent_thresholds[-1][0]).total_seconds()
                    predicted = intercept + slope * time_diff
                    
                    return max(10.0, predicted)
        
        # Fallback to pattern-based
        return self._predict_pattern_based(target_time, current_intensity)
    
    def _predict_pattern_based(
        self, 
        target_time: datetime,
        current_intensity: Optional[CarbonIntensity]
    ) -> float:
        """Predict using learned patterns."""
        return self._predict_moving_average(target_time, current_intensity)
    
    def adapt_threshold(self, recent_performance: Dict[str, Any]) -> None:
        """Adapt threshold based on recent training performance.
        
        Args:
            recent_performance: Dictionary with performance metrics
        """
        # Extract key metrics
        carbon_savings = recent_performance.get('carbon_saved_kg', 0)
        time_overhead = recent_performance.get('time_overhead_pct', 0)
        pause_frequency = recent_performance.get('pause_frequency', 0)
        
        # Calculate adaptation
        adaptation = 0.0
        
        # If saving significant carbon with acceptable time overhead, lower threshold
        if carbon_savings > 5.0 and time_overhead < 20:
            adaptation = -5.0  # Lower threshold to save more carbon
        
        # If too many pauses with little carbon savings, raise threshold
        elif pause_frequency > 0.3 and carbon_savings < 2.0:
            adaptation = 10.0  # Raise threshold to reduce interruptions
        
        # If time overhead is too high, raise threshold
        elif time_overhead > 40:
            adaptation = 15.0
        
        # Apply adaptation with learning rate
        self.adaptive_threshold += adaptation * self.learning_rate
        
        # Keep threshold within reasonable bounds
        self.adaptive_threshold = max(50.0, min(300.0, self.adaptive_threshold))
        
        # Record threshold change
        self.threshold_history.append((datetime.now(), self.adaptive_threshold))
        
        # Keep only recent history
        if len(self.threshold_history) > 50:
            self.threshold_history = self.threshold_history[-50:]
        
        logger.info(
            f"Adapted threshold: {self.adaptive_threshold:.1f} "
            f"(change: {adaptation * self.learning_rate:+.1f})"
        )
    
    async def recommend_training_schedule(
        self,
        monitor: CarbonMonitor,
        region: str,
        job_duration: timedelta,
        deadline: Optional[datetime] = None,
        required_resources: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Recommend optimal training schedule.
        
        Args:
            monitor: Carbon monitor for data
            region: Training region
            job_duration: Required training duration
            deadline: Training deadline
            required_resources: Required compute resources
            
        Returns:
            Training schedule recommendation
        """
        duration_hours = int(job_duration.total_seconds() / 3600)
        
        if not deadline:
            deadline = datetime.now() + timedelta(days=3)  # Default 3-day flexibility
        
        # Get forecast for planning window
        planning_hours = int((deadline - datetime.now()).total_seconds() / 3600)
        forecast = await monitor.get_forecast(region, hours=planning_hours)
        
        if not forecast:
            return {
                "status": "error",
                "message": "No forecast available for planning"
            }
        
        # Find optimal windows using adaptive threshold
        windows = await self._find_adaptive_windows(
            forecast, duration_hours, deadline
        )
        
        if not windows:
            return {
                "status": "no_suitable_window",
                "message": f"No suitable {duration_hours}h window found before deadline",
                "deadline": deadline.isoformat(),
                "adaptive_threshold": self.adaptive_threshold
            }
        
        best_window = windows[0]
        
        # Calculate expected savings
        baseline_emissions = self._estimate_baseline_emissions(
            duration_hours, required_resources
        )
        optimized_emissions = self._estimate_window_emissions(
            best_window, required_resources
        )
        
        carbon_savings = baseline_emissions - optimized_emissions
        
        return {
            "status": "success",
            "recommended_start": best_window.start_time.isoformat(),
            "recommended_end": best_window.end_time.isoformat(),
            "avg_carbon_intensity": best_window.avg_carbon_intensity,
            "renewable_percentage": best_window.renewable_percentage,
            "confidence_score": best_window.confidence_score,
            "carbon_savings_kg": carbon_savings,
            "delay_hours": (best_window.start_time - datetime.now()).total_seconds() / 3600,
            "adaptive_threshold": self.adaptive_threshold,
            "alternative_windows": len(windows) - 1
        }
    
    async def _find_adaptive_windows(
        self,
        forecast: Any,
        duration_hours: int,
        deadline: datetime
    ) -> List[OptimalWindow]:
        """Find optimal windows using adaptive threshold."""
        data_points = sorted(forecast.data_points, key=lambda x: x.timestamp)
        windows = []
        
        for i in range(len(data_points) - duration_hours + 1):
            window_points = data_points[i:i + duration_hours]
            
            # Check deadline constraint
            if window_points[-1].timestamp > deadline:
                break
            
            # Calculate window metrics
            avg_intensity = sum(p.carbon_intensity for p in window_points) / len(window_points)
            max_intensity = max(p.carbon_intensity for p in window_points)
            
            # Use adaptive threshold with flexibility
            effective_threshold = self.adaptive_threshold * (1 + self.workload_flexibility)
            
            if avg_intensity <= effective_threshold:
                renewable_pct = sum(
                    p.renewable_percentage or 0 
                    for p in window_points
                ) / len(window_points)
                
                # Calculate adaptive confidence score
                intensity_stability = 1.0 - (np.std([p.carbon_intensity for p in window_points]) / avg_intensity)
                threshold_margin = 1.0 - (avg_intensity / effective_threshold)
                confidence = (intensity_stability * 0.4 + threshold_margin * 0.6)
                
                window = OptimalWindow(
                    start_time=window_points[0].timestamp,
                    end_time=window_points[-1].timestamp,
                    avg_carbon_intensity=avg_intensity,
                    total_expected_carbon_kg=0.0,
                    confidence_score=max(0.1, min(1.0, confidence)),
                    renewable_percentage=renewable_pct,
                    region=forecast.region
                )
                
                windows.append(window)
        
        # Sort by composite score
        windows.sort(key=lambda w: (
            w.avg_carbon_intensity,  # Lower intensity first
            -w.renewable_percentage,  # Higher renewable first
            -w.confidence_score       # Higher confidence first
        ))
        
        return windows
    
    def _estimate_baseline_emissions(
        self, 
        duration_hours: int, 
        resources: Optional[Dict[str, Any]]
    ) -> float:
        """Estimate baseline emissions without carbon awareness."""
        # Default assumptions
        gpus = resources.get('gpus', 1) if resources else 1
        power_per_gpu_kw = resources.get('power_per_gpu_kw', 0.4) if resources else 0.4
        
        total_energy_kwh = gpus * power_per_gpu_kw * duration_hours
        
        # Assume average regional carbon intensity
        avg_carbon_intensity = 150.0  # gCO2/kWh (typical mixed grid)
        
        return total_energy_kwh * (avg_carbon_intensity / 1000)
    
    def _estimate_window_emissions(
        self, 
        window: OptimalWindow, 
        resources: Optional[Dict[str, Any]]
    ) -> float:
        """Estimate emissions for a specific training window."""
        # Default assumptions
        gpus = resources.get('gpus', 1) if resources else 1
        power_per_gpu_kw = resources.get('power_per_gpu_kw', 0.4) if resources else 0.4
        
        duration_hours = (window.end_time - window.start_time).total_seconds() / 3600
        total_energy_kwh = gpus * power_per_gpu_kw * duration_hours
        
        return total_energy_kwh * (window.avg_carbon_intensity / 1000)
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get statistics about adaptive behavior.
        
        Returns:
            Dictionary with adaptation statistics
        """
        return {
            "strategy": "adaptive",
            "workload_flexibility": self.workload_flexibility,
            "prediction_model": self.prediction_model,
            "base_threshold": self.base_threshold,
            "current_adaptive_threshold": self.adaptive_threshold,
            "threshold_adaptations": len(self.threshold_history),
            "daily_patterns_learned": len(self.daily_patterns),
            "weekly_patterns_learned": len(self.weekly_patterns),
            "historical_data_points": len(self.historical_data),
            "recent_threshold_changes": [
                {"timestamp": t[0].isoformat(), "threshold": t[1]}
                for t in self.threshold_history[-5:]
            ]
        }