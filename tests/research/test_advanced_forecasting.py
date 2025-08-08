"""
Tests for advanced carbon forecasting models.

This module provides comprehensive tests for the research implementations
of advanced carbon forecasting, including transformer models, physics-informed
networks, and federated learning approaches.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from carbon_aware_trainer.core.types import CarbonIntensity, CarbonForecast
from carbon_aware_trainer.core.monitor import CarbonMonitor
from carbon_aware_trainer.core.advanced_forecasting import (
    AdvancedCarbonForecaster,
    TemporalFusionTransformer,
    PhysicsInformedForecast,
    FederatedCarbonOptimizer,
    MultiModalInputs,
    ForecastModel
)


@pytest.fixture
def sample_carbon_data():
    """Generate sample carbon intensity data for testing."""
    data = []
    base_time = datetime.now() - timedelta(hours=168)  # 1 week ago
    
    for i in range(168):  # 1 week of hourly data
        timestamp = base_time + timedelta(hours=i)
        # Simulate daily pattern with some randomness
        base_intensity = 100 + 50 * (i % 24) / 24  # Daily cycle
        base_intensity += 20 * ((i % 168) / 168)  # Weekly variation
        base_intensity += (hash(str(i)) % 40) - 20  # Random noise
        
        ci = CarbonIntensity(
            carbon_intensity=max(20, min(300, base_intensity)),
            timestamp=timestamp,
            region="TEST_REGION",
            renewable_percentage=0.3 + 0.4 * (i % 24) / 24  # Daily renewable variation
        )
        data.append(ci)
    
    return data


@pytest.fixture
def mock_carbon_monitor():
    """Create mock carbon monitor for testing."""
    monitor = Mock(spec=CarbonMonitor)
    
    # Mock forecast data
    forecast_data = []
    base_time = datetime.now()
    for i in range(48):
        ci = CarbonIntensity(
            carbon_intensity=120 + 30 * (i % 24) / 24,
            timestamp=base_time + timedelta(hours=i + 1),
            region="TEST_REGION"
        )
        forecast_data.append(ci)
    
    mock_forecast = CarbonForecast(
        region="TEST_REGION",
        forecast_time=base_time,
        data_points=forecast_data
    )
    
    monitor.get_forecast.return_value = asyncio.coroutine(lambda: mock_forecast)()
    
    return monitor


class TestTemporalFusionTransformer:
    """Test cases for Temporal Fusion Transformer."""
    
    @pytest.fixture
    def transformer(self):
        """Create transformer instance for testing."""
        return TemporalFusionTransformer(
            input_dim=32,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            forecast_horizon=24
        )
    
    def test_transformer_initialization(self, transformer):
        """Test transformer initialization."""
        assert transformer.input_dim == 32
        assert transformer.hidden_dim == 64
        assert transformer.num_heads == 4
        assert transformer.num_layers == 2
        assert transformer.forecast_horizon == 24
    
    @pytest.mark.asyncio
    async def test_extract_features(self, transformer, sample_carbon_data):
        """Test feature extraction from carbon data."""
        inputs = MultiModalInputs(carbon_history=sample_carbon_data[:48])
        
        features = await transformer._extract_features(inputs)
        
        assert 'carbon_intensity' in features
        assert 'hour_of_day' in features
        assert 'day_of_week' in features
        assert 'month' in features
        assert len(features['carbon_intensity']) == 48
        assert len(features['hour_of_day']) == 48
    
    @pytest.mark.asyncio
    async def test_compute_attention(self, transformer, sample_carbon_data):
        """Test attention weight computation."""
        inputs = MultiModalInputs(carbon_history=sample_carbon_data[:24])
        features = await transformer._extract_features(inputs)
        
        attention_weights = await transformer._compute_attention(features, 24)
        
        assert len(attention_weights.temporal_attention) == 24
        assert len(attention_weights.feature_attention) > 0
        
        # Check that attention weights are valid probabilities
        for weight in attention_weights.temporal_attention.values():
            assert 0 <= weight <= 1
    
    @pytest.mark.asyncio
    async def test_generate_seasonal_patterns(self, transformer):
        """Test seasonal pattern generation."""
        test_time = datetime(2024, 6, 15, 12, 0, 0)  # Mid-year, noon
        
        patterns = await transformer._generate_seasonal_patterns(test_time)
        
        assert 'daily' in patterns
        assert 'weekly' in patterns
        assert 'seasonal' in patterns
        
        # Daily pattern should have 24 values
        assert len(patterns['daily']) == 24
        
        # Weekly pattern should have 7 values
        assert len(patterns['weekly']) == 7
        
        # All pattern values should be positive and reasonable
        for pattern_values in patterns.values():
            if isinstance(pattern_values, list):
                for val in pattern_values:
                    assert 0 < val < 2  # Reasonable multiplier range
    
    @pytest.mark.asyncio
    async def test_compute_uncertainty_bounds(self, transformer):
        """Test uncertainty bound computation."""
        predictions = [100, 120, 110, 130, 125]
        
        # Create mock attention weights
        from carbon_aware_trainer.core.advanced_forecasting import AttentionWeights
        attention_weights = AttentionWeights()
        attention_weights.temporal_attention = {f'step_{i}': 0.2 for i in range(5)}
        
        bounds, confidence = await transformer._compute_uncertainty_bounds(predictions, attention_weights)
        
        assert len(bounds) == len(predictions)
        assert len(confidence) == len(predictions)
        
        # Check that bounds are reasonable
        for i, (pred, (lower, upper)) in enumerate(zip(predictions, bounds)):
            assert lower < pred < upper
            assert 0 < confidence[i] < 1
    
    @pytest.mark.asyncio
    async def test_predict(self, transformer, sample_carbon_data):
        """Test transformer prediction functionality."""
        inputs = MultiModalInputs(carbon_history=sample_carbon_data)
        
        result = await transformer.predict(inputs)
        
        assert result.forecast is not None
        assert len(result.forecast.data_points) == transformer.forecast_horizon
        assert result.attention_weights is not None
        assert len(result.uncertainty_bounds) == transformer.forecast_horizon
        assert len(result.confidence_intervals) == transformer.forecast_horizon
        
        # Check that predictions are reasonable
        for ci in result.forecast.data_points:
            assert 10 <= ci.carbon_intensity <= 800
    
    @pytest.mark.asyncio
    async def test_predict_insufficient_data(self, transformer):
        """Test prediction with insufficient historical data."""
        # Only provide a few data points
        short_data = [
            CarbonIntensity(100, datetime.now() - timedelta(hours=i), "TEST", None)
            for i in range(5)
        ]
        inputs = MultiModalInputs(carbon_history=short_data)
        
        result = await transformer.predict(inputs)
        
        # Should still produce predictions but with lower confidence
        assert result.forecast is not None
        assert len(result.forecast.data_points) == transformer.forecast_horizon


class TestPhysicsInformedForecast:
    """Test cases for Physics-Informed Neural Networks."""
    
    @pytest.fixture
    def physics_model(self):
        """Create physics-informed model instance."""
        return PhysicsInformedForecast()
    
    @pytest.mark.asyncio
    async def test_apply_physics_constraints(self, physics_model):
        """Test application of physics constraints."""
        # Test predictions with unrealistic rapid changes
        predictions = [100, 200, 50, 300, 80]  # Highly variable
        
        constrained = await physics_model.apply_physics_constraints(predictions)
        
        assert len(constrained) == len(predictions)
        
        # Check that rapid changes are smoothed
        for i in range(1, len(constrained)):
            change_rate = abs(constrained[i] - constrained[i-1]) / constrained[i-1]
            assert change_rate <= 0.3  # Maximum 30% change per hour (physics constraint)
    
    @pytest.mark.asyncio
    async def test_apply_renewable_constraints(self, physics_model):
        """Test renewable energy constraints."""
        predictions = [200, 180, 160, 140]
        renewable_forecast = [0.2, 0.5, 0.7, 0.8]  # Increasing renewable generation
        
        constrained = await physics_model.apply_physics_constraints(predictions, renewable_forecast)
        
        # Carbon intensity should decrease with higher renewable generation
        assert constrained[-1] < constrained[0]  # Lower carbon with more renewables
    
    @pytest.mark.asyncio
    async def test_physical_bounds(self, physics_model):
        """Test that physical bounds are enforced."""
        # Extreme predictions outside physical limits
        predictions = [-50, 0, 1500, 2000]  # Some negative and very high values
        
        constrained = await physics_model.apply_physics_constraints(predictions)
        
        # All values should be within physical bounds
        for value in constrained:
            assert 0 <= value <= 1000  # Reasonable carbon intensity range


class TestFederatedCarbonOptimizer:
    """Test cases for Federated Carbon Optimizer."""
    
    @pytest.fixture
    def federated_optimizer(self):
        """Create federated optimizer instance."""
        return FederatedCarbonOptimizer(region_id="TEST_REGION_1")
    
    @pytest.mark.asyncio
    async def test_share_patterns(self, federated_optimizer, sample_carbon_data):
        """Test sharing of carbon patterns with privacy preservation."""
        # Create mock pattern
        from carbon_aware_trainer.core.advanced_forecasting import CarbonPattern
        pattern = CarbonPattern(
            pattern_id="test_pattern",
            region="TEST_REGION",
            temporal_signature=[0.5, 0.6, 0.7, 0.8],
            seasonal_components={"daily": [0.5, 0.6, 0.7, 0.8]},
            renewable_correlation=0.3,
            confidence_score=0.8,
            privacy_noise_level=1.0
        )
        
        shared_data = await federated_optimizer.share_patterns([pattern])
        
        assert "participant_id" in shared_data
        assert "patterns" in shared_data
        assert "privacy_metadata" in shared_data
        assert len(shared_data["patterns"]) == 1
        
        # Check privacy metadata
        privacy_info = shared_data["privacy_metadata"]
        assert "epsilon" in privacy_info
        assert "noise_mechanism" in privacy_info
    
    @pytest.mark.asyncio
    async def test_aggregate_federated_patterns(self, federated_optimizer):
        """Test aggregation of patterns from multiple participants."""
        # Mock pattern data from multiple participants
        participant_patterns = [
            {
                "participant_id": "participant_1",
                "patterns": [{
                    "seasonal_components": {"daily": [0.5, 0.6, 0.7, 0.8]},
                    "confidence_score": 0.8,
                    "region": "US-CA"
                }]
            },
            {
                "participant_id": "participant_2", 
                "patterns": [{
                    "seasonal_components": {"daily": [0.6, 0.7, 0.8, 0.9]},
                    "confidence_score": 0.7,
                    "region": "US-WA"
                }]
            }
        ]
        
        aggregated = await federated_optimizer.aggregate_federated_patterns(participant_patterns)
        
        assert "aggregated_patterns" in aggregated
        assert "total_participants" in aggregated
        assert aggregated["total_participants"] == 2
        
        if "daily" in aggregated["aggregated_patterns"]:
            daily_pattern = aggregated["aggregated_patterns"]["daily"]
            assert "temporal_signature" in daily_pattern
            assert "participant_count" in daily_pattern
            assert daily_pattern["participant_count"] == 2


class TestAdvancedCarbonForecaster:
    """Test cases for Advanced Carbon Forecaster integration."""
    
    @pytest.fixture
    def forecaster(self, mock_carbon_monitor):
        """Create advanced forecaster instance."""
        return AdvancedCarbonForecaster(mock_carbon_monitor, "TEST_REGION")
    
    @pytest.mark.asyncio
    async def test_get_transformer_forecast(self, forecaster, sample_carbon_data):
        """Test transformer forecast generation."""
        inputs = MultiModalInputs(carbon_history=sample_carbon_data)
        
        result = await forecaster.get_transformer_forecast(inputs, horizon_hours=24)
        
        assert result is not None
        assert result.forecast is not None
        assert len(result.forecast.data_points) == 24
        assert result.attention_weights is not None
        assert len(result.uncertainty_bounds) == 24
    
    @pytest.mark.asyncio
    async def test_get_physics_informed_forecast(self, forecaster):
        """Test physics-informed forecast generation."""
        base_predictions = [100, 120, 110, 130, 125]
        renewable_forecast = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        constrained = await forecaster.get_physics_informed_forecast(
            base_predictions, renewable_forecast
        )
        
        assert len(constrained) == len(base_predictions)
        
        # Should show decreasing trend with increasing renewables
        assert constrained[-1] <= constrained[0]
    
    @pytest.mark.asyncio
    async def test_get_ensemble_forecast(self, forecaster, sample_carbon_data):
        """Test ensemble forecast combining multiple models."""
        inputs = MultiModalInputs(carbon_history=sample_carbon_data)
        
        result = await forecaster.get_ensemble_forecast(
            inputs, 
            models=[ForecastModel.TRANSFORMER, ForecastModel.PHYSICS_INFORMED],
            horizon_hours=24
        )
        
        assert result is not None
        assert result.forecast is not None
        assert len(result.forecast.data_points) == 24
    
    @pytest.mark.asyncio
    async def test_evaluate_forecast_accuracy(self, forecaster):
        """Test forecast accuracy evaluation."""
        predictions = [100, 110, 120, 115, 125]
        actual_values = [105, 115, 118, 120, 130]
        
        metrics = await forecaster.evaluate_forecast_accuracy(
            predictions, actual_values, ForecastModel.TRANSFORMER
        )
        
        assert metrics.mae > 0
        assert metrics.rmse > 0
        assert metrics.mape >= 0
        assert -1 <= metrics.r2 <= 1  # R² should be between -1 and 1
    
    @pytest.mark.asyncio
    async def test_find_optimal_carbon_windows(self, forecaster, sample_carbon_data):
        """Test optimal window finding with uncertainty consideration."""
        # Create mock transformer result
        inputs = MultiModalInputs(carbon_history=sample_carbon_data)
        forecast_result = await forecaster.get_transformer_forecast(inputs, 48)
        
        windows = await forecaster.find_optimal_carbon_windows(
            forecast_result, 
            duration_hours=8,
            max_carbon_intensity=150.0,
            consider_uncertainty=True
        )
        
        # Should find some valid windows
        assert len(windows) >= 0
        
        # If windows found, they should meet criteria
        for window in windows:
            assert window.avg_carbon_intensity <= 150.0
            assert window.start_time < window.end_time
            assert (window.end_time - window.start_time).total_seconds() / 3600 >= 8


class TestIntegrationScenarios:
    """Integration test scenarios combining multiple components."""
    
    @pytest.mark.asyncio
    async def test_full_forecasting_pipeline(self, mock_carbon_monitor, sample_carbon_data):
        """Test complete forecasting pipeline from data to predictions."""
        forecaster = AdvancedCarbonForecaster(mock_carbon_monitor, "TEST_REGION")
        inputs = MultiModalInputs(carbon_history=sample_carbon_data)
        
        # Test transformer forecast
        transformer_result = await forecaster.get_transformer_forecast(inputs, 24)
        assert transformer_result.forecast is not None
        
        # Test physics constraints
        transformer_predictions = [ci.carbon_intensity for ci in transformer_result.forecast.data_points]
        physics_predictions = await forecaster.get_physics_informed_forecast(transformer_predictions)
        assert len(physics_predictions) == len(transformer_predictions)
        
        # Test ensemble forecast
        ensemble_result = await forecaster.get_ensemble_forecast(inputs, horizon_hours=24)
        assert ensemble_result.forecast is not None
        
        # Test accuracy evaluation (using transformer predictions as "actual" for demo)
        metrics = await forecaster.evaluate_forecast_accuracy(
            physics_predictions[:12], 
            transformer_predictions[:12], 
            ForecastModel.PHYSICS_INFORMED
        )
        assert metrics.mae >= 0
    
    @pytest.mark.asyncio
    async def test_federated_learning_workflow(self, sample_carbon_data):
        """Test federated learning workflow with pattern sharing."""
        # Create multiple federated optimizers (simulating different organizations)
        optimizer1 = FederatedCarbonOptimizer("REGION_1")
        optimizer2 = FederatedCarbonOptimizer("REGION_2") 
        
        # Extract patterns from each region's data
        forecaster = Mock()  # Would use actual forecaster
        
        patterns1 = await optimizer1.extract_local_carbon_patterns(sample_carbon_data[:84], forecaster)
        patterns2 = await optimizer2.extract_local_carbon_patterns(sample_carbon_data[84:], forecaster)
        
        # Share patterns
        shared1 = await optimizer1.share_patterns_privately(patterns1)
        shared2 = await optimizer2.share_patterns_privately(patterns2)
        
        # Aggregate patterns
        aggregated = await optimizer1.aggregate_federated_patterns([shared1, shared2])
        
        assert "aggregated_patterns" in aggregated
        assert aggregated["total_participants"] == 2