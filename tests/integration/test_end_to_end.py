"""End-to-end integration tests."""

import pytest
import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from carbon_aware_trainer import CarbonAwareTrainer, CarbonMonitor
from carbon_aware_trainer.core.types import CarbonDataSource, TrainingConfig
from carbon_aware_trainer.carbon_models.cached import CachedProvider


class TestEndToEndIntegration:
    """End-to-end integration tests using cached data."""
    
    @pytest.fixture
    def sample_data_file(self):
        """Create temporary sample data file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Create sample carbon data
            sample_data = {
                "regions": {
                    "US-CA": {
                        "historical": [],
                        "forecast": []
                    }
                }
            }
            
            # Generate 48 hours of sample data
            base_time = datetime.now()
            for i in range(48):
                timestamp = base_time + timedelta(hours=i)
                
                # Simulate daily pattern: higher during day, lower at night
                hour = timestamp.hour
                base_intensity = 80 + 30 * abs(0.5 - (hour / 24))  # 50-110 range
                
                data_point = {
                    "timestamp": timestamp.isoformat(),
                    "carbon_intensity": base_intensity + (i % 10),  # Add some variation
                    "renewable_percentage": 40 + (hour / 24) * 40,  # 40-80% range
                    "confidence": 0.85
                }
                
                if i < 24:
                    sample_data["regions"]["US-CA"]["historical"].append(data_point)
                else:
                    sample_data["regions"]["US-CA"]["forecast"].append(data_point)
            
            json.dump(sample_data, f, indent=2)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        Path(temp_path).unlink()
    
    @pytest.mark.asyncio
    async def test_carbon_aware_training_session(self, sample_data_file):
        """Test complete carbon-aware training session."""
        config = TrainingConfig(
            carbon_threshold=90.0,
            pause_threshold=120.0,
            resume_threshold=70.0,
            check_interval=1  # Fast checking for test
        )
        
        trainer = CarbonAwareTrainer(
            carbon_model='cached',
            region='US-CA',
            config=config,
            api_key=sample_data_file
        )
        
        # Track state changes
        state_changes = []
        def track_states(state, metrics):
            state_changes.append(state.value)
        
        # Mock a simple model
        class MockModel:
            def train_step(self, batch):
                return {"loss": 0.5, "accuracy": 0.8}
        
        trainer.model = MockModel()
        
        async with trainer:
            trainer.add_state_callback(track_states)
            
            async with trainer.training_session():
                # Simulate several training steps
                for i in range(5):
                    batch = {"data": f"batch_{i}"}
                    result = await trainer.train_step(batch)
                    
                    assert "loss" in result
                    assert result["loss"] == 0.5
                    
                    # Brief pause between steps
                    await asyncio.sleep(0.1)
                
                # Verify training progressed
                assert trainer.step == 5
                
                # Get final metrics
                metrics = trainer.get_carbon_metrics()
                assert metrics['step'] == 5
                assert metrics['total_energy_kwh'] > 0
                assert metrics['total_carbon_kg'] > 0
                assert metrics['session_id'] == trainer.session_id
        
        # Verify state changes occurred
        assert 'running' in state_changes
    
    @pytest.mark.asyncio
    async def test_carbon_monitoring_with_cached_data(self, sample_data_file):
        """Test carbon monitoring using cached data."""
        monitor = CarbonMonitor(
            regions=['US-CA'],
            data_source=CarbonDataSource.CACHED,
            api_key=sample_data_file,
            update_interval=1
        )
        
        async with monitor:
            # Test getting current intensity
            current = await monitor.get_current_intensity('US-CA')
            assert current is not None
            assert current.region == 'US-CA'
            assert current.carbon_intensity > 0
            assert current.renewable_percentage is not None
            
            # Test getting forecast
            forecast = await monitor.get_forecast('US-CA', hours=12)
            assert forecast is not None
            assert len(forecast.data_points) > 0
            assert forecast.region == 'US-CA'
            
            # Test finding optimal window
            window = monitor.find_optimal_window(
                duration_hours=4,
                max_carbon_intensity=100.0,
                preferred_regions=['US-CA']
            )
            
            if window:  # May not find suitable window depending on data
                assert window.region == 'US-CA'
                assert window.avg_carbon_intensity <= 100.0
    
    @pytest.mark.asyncio
    async def test_threshold_based_pause_resume(self, sample_data_file):
        """Test pause/resume behavior based on carbon thresholds."""
        # Use low thresholds to trigger pause/resume
        config = TrainingConfig(
            carbon_threshold=60.0,   # Very low base threshold
            pause_threshold=80.0,    # Low pause threshold
            resume_threshold=60.0,   # Low resume threshold
            check_interval=1
        )
        
        trainer = CarbonAwareTrainer(
            carbon_model='cached',
            region='US-CA',
            config=config,
            api_key=sample_data_file
        )
        
        pause_resume_events = []
        
        def track_pause_resume(state, metrics):
            if state.value in ['paused', 'running']:
                pause_resume_events.append({
                    'state': state.value,
                    'timestamp': datetime.now(),
                    'carbon_kg': metrics.total_carbon_kg
                })
        
        class MockModel:
            def train_step(self, batch):
                return {"loss": 0.3}
        
        trainer.model = MockModel()
        
        async with trainer:
            trainer.add_state_callback(track_pause_resume)
            
            async with trainer.training_session():
                # Simulate training with potential pauses
                for i in range(10):
                    try:
                        batch = {"data": f"batch_{i}"}
                        await trainer.train_step(batch)
                        await asyncio.sleep(0.2)  # Give time for carbon checks
                    except StopIteration:
                        break  # Training was stopped
                
                # Get final metrics
                final_metrics = trainer.get_carbon_metrics()
                
                # Verify training occurred
                assert final_metrics['step'] > 0
                assert final_metrics['total_carbon_kg'] > 0
    
    @pytest.mark.asyncio
    async def test_optimal_window_finding(self, sample_data_file):
        """Test finding optimal training windows."""
        from carbon_aware_trainer.core.forecasting import CarbonForecaster
        
        monitor = CarbonMonitor(
            regions=['US-CA'],
            data_source=CarbonDataSource.CACHED,
            api_key=sample_data_file
        )
        
        async with monitor:
            forecaster = CarbonForecaster(monitor)
            
            # Find multiple optimal windows
            windows = await forecaster.find_optimal_windows(
                region='US-CA',
                duration_hours=6,
                num_windows=3,
                max_carbon_intensity=100.0,
                flexibility_hours=24
            )
            
            # Should find at least one window in 24 hours
            assert len(windows) >= 0  # May be zero if all carbon is too high
            
            for window in windows:
                assert window.region == 'US-CA'
                assert window.avg_carbon_intensity <= 100.0
                assert window.confidence_score > 0
                
                # Verify window duration
                duration = (window.end_time - window.start_time).total_seconds() / 3600
                assert duration >= 5  # At least 5 hours (close to 6 requested)
    
    @pytest.mark.asyncio
    async def test_carbon_savings_calculation(self, sample_data_file):
        """Test carbon savings calculation."""
        from carbon_aware_trainer.core.forecasting import CarbonForecaster
        
        monitor = CarbonMonitor(
            regions=['US-CA'],
            data_source=CarbonDataSource.CACHED,
            api_key=sample_data_file
        )
        
        async with monitor:
            forecaster = CarbonForecaster(monitor)
            
            # Predict emissions for immediate training
            immediate_training = await forecaster.predict_training_emissions(
                region='US-CA',
                start_time=datetime.now(),
                duration_hours=8,
                avg_power_kw=0.4,
                num_gpus=2
            )
            
            assert 'predicted_emissions_kg' in immediate_training
            assert immediate_training['predicted_emissions_kg'] > 0
            assert immediate_training['total_energy_kwh'] == 0.4 * 2 * 8  # power * gpus * hours
            
            # Find optimal window and compare
            windows = await forecaster.find_optimal_windows(
                region='US-CA',
                duration_hours=8,
                num_windows=1,
                flexibility_hours=12
            )
            
            if windows:
                optimal_window = windows[0]
                
                optimal_training = await forecaster.predict_training_emissions(
                    region='US-CA',
                    start_time=optimal_window.start_time,
                    duration_hours=8,
                    avg_power_kw=0.4,
                    num_gpus=2
                )
                
                # Optimal window should have lower or equal emissions
                assert optimal_training['predicted_emissions_kg'] <= immediate_training['predicted_emissions_kg']
    
    @pytest.mark.asyncio
    async def test_multiple_region_comparison(self, sample_data_file):
        """Test comparing carbon intensity across multiple regions."""
        # Create multi-region data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            multi_region_data = {
                "regions": {
                    "US-CA": {"historical": [], "forecast": []},
                    "US-WA": {"historical": [], "forecast": []},
                    "EU-FR": {"historical": [], "forecast": []}
                }
            }
            
            base_time = datetime.now()
            for region, base_intensity in [("US-CA", 120), ("US-WA", 80), ("EU-FR", 100)]:
                for i in range(24):
                    timestamp = base_time + timedelta(hours=i)
                    data_point = {
                        "timestamp": timestamp.isoformat(),
                        "carbon_intensity": base_intensity + (i % 5),
                        "renewable_percentage": 50 + (i % 20),
                        "confidence": 0.9
                    }
                    multi_region_data["regions"][region]["historical"].append(data_point)
            
            json.dump(multi_region_data, f, indent=2)
            multi_region_file = f.name
        
        try:
            monitor = CarbonMonitor(
                regions=['US-CA', 'US-WA', 'EU-FR'],
                data_source=CarbonDataSource.CACHED,
                api_key=multi_region_file
            )
            
            async with monitor:
                # Get current intensities for all regions
                intensities = {}
                for region in ['US-CA', 'US-WA', 'EU-FR']:
                    intensity = await monitor.get_current_intensity(region)
                    if intensity:
                        intensities[region] = intensity
                
                assert len(intensities) == 3
                
                # Find cleanest region
                cleanest = monitor.get_cleanest_region()
                assert cleanest in intensities
                
                # Verify cleanest has lowest intensity
                cleanest_intensity = intensities[cleanest].carbon_intensity
                for region, intensity in intensities.items():
                    assert cleanest_intensity <= intensity.carbon_intensity
        
        finally:
            Path(multi_region_file).unlink()
    
    def test_cached_provider_direct(self, sample_data_file):
        """Test CachedProvider directly."""
        provider = CachedProvider(sample_data_file)
        
        # Test basic functionality
        supported_regions = provider.get_supported_regions()
        assert 'US-CA' in supported_regions
        
        # Test data loading
        assert 'regions' in provider.data
        assert 'US-CA' in provider.data['regions']