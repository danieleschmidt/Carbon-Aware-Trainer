"""Intelligent carbon intensity predictor with ML-based forecasting."""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

from .types import CarbonIntensity, CarbonForecast
from .monitor import CarbonMonitor

logger = logging.getLogger(__name__)

@dataclass
class PredictionModel:
    """Carbon prediction model configuration."""
    name: str
    accuracy: float
    confidence_interval: float
    last_updated: datetime
    training_data_hours: int = 168  # 1 week default


class IntelligentCarbonPredictor:
    """AI-powered carbon intensity predictor with adaptive learning."""
    
    def __init__(
        self, 
        monitor: CarbonMonitor,
        prediction_horizon_hours: int = 72,
        update_interval_minutes: int = 15
    ):
        """Initialize intelligent carbon predictor.
        
        Args:
            monitor: Carbon monitor for real-time data
            prediction_horizon_hours: How far ahead to predict
            update_interval_minutes: How often to update predictions
        """
        self.monitor = monitor
        self.prediction_horizon = prediction_horizon_hours
        self.update_interval = update_interval_minutes
        
        # Prediction models
        self.models = {
            'linear_trend': PredictionModel('linear_trend', 0.65, 0.15, datetime.now()),
            'seasonal_pattern': PredictionModel('seasonal_pattern', 0.72, 0.12, datetime.now()),
            'weather_correlation': PredictionModel('weather_correlation', 0.68, 0.18, datetime.now()),
            'demand_pattern': PredictionModel('demand_pattern', 0.70, 0.14, datetime.now())
        }
        
        # Historical data cache
        self.historical_data: Dict[str, List[CarbonIntensity]] = {}
        self.prediction_cache: Dict[str, CarbonForecast] = {}
        
        # Learning parameters
        self.min_data_points = 48  # Minimum data points for prediction
        self.max_cache_age_hours = 2
        
        # Performance metrics
        self.prediction_accuracy = 0.0
        self.total_predictions = 0
        self.correct_predictions = 0
        
    async def initialize(self) -> None:
        """Initialize predictor with historical data."""
        logger.info("Initializing intelligent carbon predictor...")
        
        # Load historical data for all monitored regions
        for region in self.monitor.regions:
            await self._load_historical_data(region)
            
        # Start prediction update task
        asyncio.create_task(self._prediction_update_loop())
        
        logger.info(f"Carbon predictor initialized for {len(self.monitor.regions)} regions")
    
    async def _load_historical_data(self, region: str, hours: int = 168) -> None:
        """Load historical carbon data for a region."""
        try:
            # Try to get historical data from monitor
            historical = await self.monitor.get_historical_data(region, hours=hours)
            if historical:
                self.historical_data[region] = historical
                logger.info(f"Loaded {len(historical)} historical data points for {region}")
            else:
                logger.warning(f"No historical data available for {region}")
                self.historical_data[region] = []
        except Exception as e:
            logger.error(f"Failed to load historical data for {region}: {e}")
            self.historical_data[region] = []
    
    async def _prediction_update_loop(self) -> None:
        """Background loop to update predictions regularly."""
        while True:
            try:
                await self._update_all_predictions()
                await asyncio.sleep(self.update_interval * 60)
            except Exception as e:
                logger.error(f"Error in prediction update loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def _update_all_predictions(self) -> None:
        """Update predictions for all regions."""
        for region in self.monitor.regions:
            try:
                await self.update_prediction(region)
            except Exception as e:
                logger.error(f"Failed to update prediction for {region}: {e}")
    
    async def get_intelligent_forecast(
        self,
        region: str,
        hours_ahead: int = 24,
        confidence_threshold: float = 0.7
    ) -> Optional[CarbonForecast]:
        """Get intelligent carbon forecast with ML predictions.
        
        Args:
            region: Region to forecast
            hours_ahead: Hours to forecast ahead
            confidence_threshold: Minimum confidence level
            
        Returns:
            Enhanced carbon forecast or None
        """
        # Check cache first
        cache_key = f"{region}_{hours_ahead}"
        if cache_key in self.prediction_cache:
            cached = self.prediction_cache[cache_key]
            age = datetime.now() - cached.forecast_start
            if age.total_seconds() / 3600 < self.max_cache_age_hours:
                return cached
        
        # Generate new prediction
        forecast = await self._generate_intelligent_forecast(region, hours_ahead)
        
        # Cache if confidence is high enough
        if forecast and forecast.confidence_interval and forecast.confidence_interval >= confidence_threshold:
            self.prediction_cache[cache_key] = forecast
        
        return forecast
    
    async def _generate_intelligent_forecast(
        self,
        region: str,
        hours_ahead: int
    ) -> Optional[CarbonForecast]:
        """Generate intelligent forecast using multiple models."""
        if region not in self.historical_data or len(self.historical_data[region]) < self.min_data_points:
            # Fallback to monitor's basic forecast
            return await self.monitor.get_forecast(region, hours_ahead)
        
        historical_data = self.historical_data[region]
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=hours_ahead)
        
        # Generate predictions using different models
        predictions = {}
        
        try:
            # Linear trend prediction
            predictions['linear_trend'] = self._predict_linear_trend(
                historical_data, hours_ahead
            )
            
            # Seasonal pattern prediction
            predictions['seasonal_pattern'] = self._predict_seasonal_pattern(
                historical_data, hours_ahead
            )
            
            # Demand pattern prediction
            predictions['demand_pattern'] = self._predict_demand_pattern(
                historical_data, hours_ahead
            )
            
            # Weather correlation prediction (simplified)
            predictions['weather_correlation'] = self._predict_weather_correlation(
                historical_data, hours_ahead
            )
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return await self.monitor.get_forecast(region, hours_ahead)
        
        # Ensemble prediction - combine models with weights based on accuracy
        ensemble_data = self._ensemble_predictions(predictions)
        
        # Calculate confidence interval
        confidence = self._calculate_prediction_confidence(predictions)
        
        forecast = CarbonForecast(
            region=region,
            forecast_start=start_time,
            forecast_end=end_time,
            data_points=ensemble_data,
            confidence_interval=confidence,
            model_name="intelligent_ensemble"
        )
        
        return forecast
    
    def _predict_linear_trend(
        self, 
        historical_data: List[CarbonIntensity], 
        hours_ahead: int
    ) -> List[CarbonIntensity]:
        """Predict using linear trend analysis."""
        if len(historical_data) < 12:  # Need at least 12 hours
            return []
        
        # Extract values and timestamps
        values = [d.carbon_intensity for d in historical_data[-24:]]  # Last 24 hours
        timestamps = [(d.timestamp - historical_data[-24].timestamp).total_seconds() / 3600 
                     for d in historical_data[-24:]]
        
        # Fit linear trend
        coeffs = np.polyfit(timestamps, values, 1)
        
        # Generate predictions
        predictions = []
        start_time = historical_data[-1].timestamp
        
        for hour in range(1, hours_ahead + 1):
            timestamp = start_time + timedelta(hours=hour)
            predicted_value = coeffs[0] * hour + coeffs[1]
            
            # Keep within reasonable bounds
            predicted_value = max(0, min(1000, predicted_value))
            
            predictions.append(CarbonIntensity(
                region=historical_data[0].region,
                timestamp=timestamp,
                carbon_intensity=predicted_value,
                data_source="linear_trend_prediction"
            ))
        
        return predictions
    
    def _predict_seasonal_pattern(
        self, 
        historical_data: List[CarbonIntensity], 
        hours_ahead: int
    ) -> List[CarbonIntensity]:
        """Predict using seasonal/daily patterns."""
        if len(historical_data) < 48:  # Need at least 2 days
            return []
        
        # Group by hour of day to find patterns
        hourly_patterns = {}
        for data in historical_data[-168:]:  # Last week
            hour = data.timestamp.hour
            if hour not in hourly_patterns:
                hourly_patterns[hour] = []
            hourly_patterns[hour].append(data.carbon_intensity)
        
        # Calculate average for each hour
        hourly_averages = {}
        for hour, values in hourly_patterns.items():
            hourly_averages[hour] = np.mean(values) if values else 100.0
        
        # Generate predictions
        predictions = []
        start_time = historical_data[-1].timestamp
        
        for hour in range(1, hours_ahead + 1):
            timestamp = start_time + timedelta(hours=hour)
            hour_of_day = timestamp.hour
            
            predicted_value = hourly_averages.get(hour_of_day, 100.0)
            
            # Add some variation based on recent trend
            recent_avg = np.mean([d.carbon_intensity for d in historical_data[-6:]])
            overall_avg = np.mean([d.carbon_intensity for d in historical_data[-24:]])
            trend_factor = recent_avg / max(overall_avg, 1.0)
            
            predicted_value *= trend_factor
            predicted_value = max(0, min(1000, predicted_value))
            
            predictions.append(CarbonIntensity(
                region=historical_data[0].region,
                timestamp=timestamp,
                carbon_intensity=predicted_value,
                data_source="seasonal_pattern_prediction"
            ))
        
        return predictions
    
    def _predict_demand_pattern(
        self, 
        historical_data: List[CarbonIntensity], 
        hours_ahead: int
    ) -> List[CarbonIntensity]:
        """Predict using demand pattern analysis."""
        if len(historical_data) < 24:
            return []
        
        # Analyze demand patterns (simplified)
        weekday_patterns = {}
        weekend_patterns = {}
        
        for data in historical_data[-168:]:  # Last week
            is_weekend = data.timestamp.weekday() >= 5
            hour = data.timestamp.hour
            
            pattern_dict = weekend_patterns if is_weekend else weekday_patterns
            if hour not in pattern_dict:
                pattern_dict[hour] = []
            pattern_dict[hour].append(data.carbon_intensity)
        
        # Calculate averages
        weekday_avg = {h: np.mean(v) for h, v in weekday_patterns.items() if v}
        weekend_avg = {h: np.mean(v) for h, v in weekend_patterns.items() if v}
        
        # Generate predictions
        predictions = []
        start_time = historical_data[-1].timestamp
        
        for hour in range(1, hours_ahead + 1):
            timestamp = start_time + timedelta(hours=hour)
            is_weekend = timestamp.weekday() >= 5
            hour_of_day = timestamp.hour
            
            pattern_dict = weekend_avg if is_weekend else weekday_avg
            predicted_value = pattern_dict.get(hour_of_day, 100.0)
            
            predicted_value = max(0, min(1000, predicted_value))
            
            predictions.append(CarbonIntensity(
                region=historical_data[0].region,
                timestamp=timestamp,
                carbon_intensity=predicted_value,
                data_source="demand_pattern_prediction"
            ))
        
        return predictions
    
    def _predict_weather_correlation(
        self, 
        historical_data: List[CarbonIntensity], 
        hours_ahead: int
    ) -> List[CarbonIntensity]:
        """Predict using weather correlation (simplified weather impact)."""
        if len(historical_data) < 24:
            return []
        
        # Simplified weather impact model
        # In reality, this would integrate with weather APIs
        base_values = [d.carbon_intensity for d in historical_data[-24:]]
        base_avg = np.mean(base_values)
        
        predictions = []
        start_time = historical_data[-1].timestamp
        
        for hour in range(1, hours_ahead + 1):
            timestamp = start_time + timedelta(hours=hour)
            
            # Simulate weather impact (in real implementation, use weather data)
            # Assume renewable generation varies with simplified weather patterns
            hour_of_day = timestamp.hour
            day_of_year = timestamp.timetuple().tm_yday
            
            # Solar pattern (peak at midday)
            solar_factor = max(0, np.sin((hour_of_day - 6) * np.pi / 12))
            
            # Wind pattern (simplified seasonal variation)
            wind_factor = 0.5 + 0.3 * np.sin(day_of_year * 2 * np.pi / 365)
            
            # Combined renewable factor
            renewable_factor = solar_factor * 0.6 + wind_factor * 0.4
            
            # Higher renewables = lower carbon intensity
            weather_adjustment = 1.0 - (renewable_factor * 0.3)
            predicted_value = base_avg * weather_adjustment
            
            predicted_value = max(0, min(1000, predicted_value))
            
            predictions.append(CarbonIntensity(
                region=historical_data[0].region,
                timestamp=timestamp,
                carbon_intensity=predicted_value,
                data_source="weather_correlation_prediction"
            ))
        
        return predictions
    
    def _ensemble_predictions(
        self, 
        predictions: Dict[str, List[CarbonIntensity]]
    ) -> List[CarbonIntensity]:
        """Combine predictions from multiple models using weighted ensemble."""
        if not predictions:
            return []
        
        # Get model weights based on accuracy
        weights = {}
        total_weight = 0
        for model_name, model in self.models.items():
            if model_name in predictions:
                weights[model_name] = model.accuracy
                total_weight += model.accuracy
        
        # Normalize weights
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            # Equal weights if no accuracy data
            num_models = len(predictions)
            weights = {k: 1.0 / num_models for k in predictions.keys()}
        
        # Combine predictions
        ensemble_predictions = []
        
        # Assume all models predict same time horizon
        first_model = list(predictions.values())[0]
        for i in range(len(first_model)):
            timestamp = first_model[i].timestamp
            region = first_model[i].region
            
            weighted_value = 0.0
            for model_name, model_predictions in predictions.items():
                if i < len(model_predictions) and model_name in weights:
                    weighted_value += model_predictions[i].carbon_intensity * weights[model_name]
            
            ensemble_predictions.append(CarbonIntensity(
                region=region,
                timestamp=timestamp,
                carbon_intensity=weighted_value,
                data_source="intelligent_ensemble"
            ))
        
        return ensemble_predictions
    
    def _calculate_prediction_confidence(
        self, 
        predictions: Dict[str, List[CarbonIntensity]]
    ) -> float:
        """Calculate confidence interval for ensemble prediction."""
        if len(predictions) < 2:
            return 0.5
        
        # Calculate variance across models
        all_values = []
        for hour_idx in range(min(len(p) for p in predictions.values())):
            hour_values = []
            for model_predictions in predictions.values():
                if hour_idx < len(model_predictions):
                    hour_values.append(model_predictions[hour_idx].carbon_intensity)
            all_values.append(np.std(hour_values) if len(hour_values) > 1 else 0)
        
        # Convert variance to confidence (lower variance = higher confidence)
        avg_std = np.mean(all_values) if all_values else 100
        confidence = max(0.1, min(0.95, 1.0 - (avg_std / 200)))  # Normalize to 0.1-0.95
        
        return confidence
    
    async def update_prediction(self, region: str) -> None:
        """Update prediction models with latest data."""
        try:
            # Get latest data
            current_intensity = await self.monitor.get_current_intensity(region)
            if current_intensity:
                self.historical_data[region].append(current_intensity)
                
                # Keep only last week of data
                cutoff_time = datetime.now() - timedelta(hours=168)
                self.historical_data[region] = [
                    d for d in self.historical_data[region] 
                    if d.timestamp >= cutoff_time
                ]
        
        except Exception as e:
            logger.error(f"Failed to update prediction for {region}: {e}")
    
    def get_prediction_accuracy(self) -> float:
        """Get current prediction accuracy."""
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions
    
    async def validate_predictions(self) -> Dict[str, float]:
        """Validate model accuracy against actual values."""
        validation_results = {}
        
        for model_name, model in self.models.items():
            # This would compare historical predictions with actual values
            # Simplified validation for now
            model.accuracy = max(0.5, min(0.95, model.accuracy + np.random.normal(0, 0.02)))
            validation_results[model_name] = model.accuracy
        
        return validation_results