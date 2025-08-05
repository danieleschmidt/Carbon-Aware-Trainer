"""Unit tests for carbon monitor."""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from carbon_aware_trainer.core.monitor import CarbonMonitor
from carbon_aware_trainer.core.types import (
    CarbonIntensity, CarbonForecast, CarbonDataSource,
    CarbonIntensityUnit
)


class TestCarbonMonitor:
    """Test cases for CarbonMonitor."""
    
    @pytest.fixture
    def sample_intensity(self):
        """Sample carbon intensity data."""
        return CarbonIntensity(
            region="US-CA",
            timestamp=datetime.now(),
            carbon_intensity=120.0,
            unit=CarbonIntensityUnit.GRAMS_CO2_PER_KWH,
            data_source="test",
            renewable_percentage=45.0
        )
    
    @pytest.fixture
    def sample_forecast(self):
        """Sample carbon forecast data."""
        data_points = []
        base_time = datetime.now()
        
        for i in range(24):  # 24 hours of data
            data_points.append(CarbonIntensity(
                region="US-CA",
                timestamp=base_time + timedelta(hours=i),
                carbon_intensity=100.0 + (i * 2),  # Increasing intensity
                unit=CarbonIntensityUnit.GRAMS_CO2_PER_KWH,
                data_source="test"
            ))
        
        return CarbonForecast(
            region="US-CA",
            forecast_start=base_time,
            forecast_end=base_time + timedelta(hours=24),
            data_points=data_points,
            model_name="test"
        )
    
    @pytest.fixture
    def mock_provider(self):
        """Mock carbon data provider."""
        provider = AsyncMock()
        provider.__aenter__ = AsyncMock(return_value=provider)
        provider.__aexit__ = AsyncMock(return_value=None)
        return provider
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        regions = ["US-CA", "EU-FR"]
        monitor = CarbonMonitor(
            regions=regions,
            data_source=CarbonDataSource.ELECTRICITYMAP,
            update_interval=300
        )
        
        assert monitor.regions == regions
        assert monitor.data_source == CarbonDataSource.ELECTRICITYMAP
        assert monitor.update_interval == 300
        assert monitor._current_intensities == {}
        assert monitor._forecasts == {}
    
    @pytest.mark.asyncio
    async def test_monitor_context_manager(self, mock_provider):
        """Test monitor as async context manager."""
        with patch('carbon_aware_trainer.core.monitor.ElectricityMapProvider', return_value=mock_provider):
            monitor = CarbonMonitor(
                regions=["US-CA"],
                data_source=CarbonDataSource.ELECTRICITYMAP
            )
            
            async with monitor:
                assert monitor.provider is not None
                mock_provider.__aenter__.assert_called_once()
            
            mock_provider.__aexit__.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, mock_provider):
        """Test starting and stopping monitoring."""
        with patch('carbon_aware_trainer.core.monitor.ElectricityMapProvider', return_value=mock_provider):
            monitor = CarbonMonitor(
                regions=["US-CA"],
                data_source=CarbonDataSource.ELECTRICITYMAP,
                update_interval=1  # 1 second for fast testing
            )
            
            async with monitor:
                await monitor.start_monitoring()
                assert monitor._monitoring_task is not None
                assert not monitor._monitoring_task.done()
                
                await asyncio.sleep(0.1)  # Let monitoring loop run briefly
                
                await monitor.stop_monitoring()
                assert monitor._stop_monitoring is True
    
    @pytest.mark.asyncio
    async def test_get_current_intensity(self, mock_provider, sample_intensity):
        """Test getting current carbon intensity."""
        mock_provider.get_current_intensity.return_value = sample_intensity
        
        with patch('carbon_aware_trainer.core.monitor.ElectricityMapProvider', return_value=mock_provider):
            monitor = CarbonMonitor(
                regions=["US-CA"],
                data_source=CarbonDataSource.ELECTRICITYMAP
            )
            
            async with monitor:
                intensity = await monitor.get_current_intensity("US-CA")
                
                assert intensity is not None
                assert intensity.carbon_intensity == 120.0
                assert intensity.region == "US-CA"
                mock_provider.get_current_intensity.assert_called_with("US-CA")
    
    @pytest.mark.asyncio
    async def test_get_forecast(self, mock_provider, sample_forecast):
        """Test getting carbon forecast."""
        mock_provider.get_forecast.return_value = sample_forecast
        
        with patch('carbon_aware_trainer.core.monitor.ElectricityMapProvider', return_value=mock_provider):
            monitor = CarbonMonitor(
                regions=["US-CA"],
                data_source=CarbonDataSource.ELECTRICITYMAP
            )
            
            async with monitor:
                forecast = await monitor.get_forecast("US-CA", hours=24)
                
                assert forecast is not None
                assert len(forecast.data_points) == 24
                assert forecast.region == "US-CA"
                mock_provider.get_forecast.assert_called_once()
    
    def test_find_optimal_window(self, sample_forecast):
        """Test finding optimal training window."""
        monitor = CarbonMonitor(
            regions=["US-CA"],
            data_source=CarbonDataSource.ELECTRICITYMAP
        )
        
        # Add forecast to cache
        monitor._forecasts["US-CA"] = sample_forecast
        
        # Find 8-hour window with max 130 gCO2/kWh
        window = monitor.find_optimal_window(
            duration_hours=8,
            max_carbon_intensity=130.0,
            preferred_regions=["US-CA"]
        )
        
        assert window is not None
        assert window.region == "US-CA"
        assert window.avg_carbon_intensity <= 130.0
        assert (window.end_time - window.start_time).total_seconds() / 3600 >= 7  # At least 7 hours
    
    def test_find_optimal_window_no_suitable(self, sample_forecast):
        """Test finding optimal window when none suitable."""
        monitor = CarbonMonitor(
            regions=["US-CA"],
            data_source=CarbonDataSource.ELECTRICITYMAP
        )
        
        # Add forecast to cache
        monitor._forecasts["US-CA"] = sample_forecast
        
        # Try to find window with very low threshold (should fail)
        window = monitor.find_optimal_window(
            duration_hours=8,
            max_carbon_intensity=50.0,  # Very low threshold
            preferred_regions=["US-CA"]
        )
        
        assert window is None
    
    @pytest.mark.asyncio
    async def test_carbon_change_callback(self, mock_provider):
        """Test carbon intensity change callback."""
        callback_calls = []
        
        def test_callback(event_type, data):
            callback_calls.append((event_type, data))
        
        with patch('carbon_aware_trainer.core.monitor.ElectricityMapProvider', return_value=mock_provider):
            monitor = CarbonMonitor(
                regions=["US-CA"],
                data_source=CarbonDataSource.ELECTRICITYMAP
            )
            
            monitor.add_callback(test_callback)
            
            # Simulate intensity change
            old_intensity = CarbonIntensity(
                region="US-CA",
                timestamp=datetime.now(),
                carbon_intensity=100.0,
                unit=CarbonIntensityUnit.GRAMS_CO2_PER_KWH,
                data_source="test"
            )
            
            new_intensity = CarbonIntensity(
                region="US-CA",
                timestamp=datetime.now(),
                carbon_intensity=150.0,  # Significant change
                unit=CarbonIntensityUnit.GRAMS_CO2_PER_KWH,
                data_source="test"
            )
            
            monitor._current_intensities["US-CA"] = old_intensity
            
            await monitor._notify_callbacks('intensity_change', {
                'region': 'US-CA',
                'old_intensity': old_intensity,
                'new_intensity': new_intensity
            })
            
            assert len(callback_calls) == 1
            assert callback_calls[0][0] == 'intensity_change'
            assert callback_calls[0][1]['region'] == 'US-CA'
    
    def test_get_cleanest_region(self, mock_provider):
        """Test finding cleanest region."""
        with patch('carbon_aware_trainer.core.monitor.ElectricityMapProvider', return_value=mock_provider):
            monitor = CarbonMonitor(
                regions=["US-CA", "EU-FR"],
                data_source=CarbonDataSource.ELECTRICITYMAP
            )
            
            # Set up current intensities
            monitor._current_intensities = {
                "US-CA": CarbonIntensity(
                    region="US-CA",
                    timestamp=datetime.now(),
                    carbon_intensity=150.0,
                    unit=CarbonIntensityUnit.GRAMS_CO2_PER_KWH,
                    data_source="test"
                ),
                "EU-FR": CarbonIntensity(
                    region="EU-FR",
                    timestamp=datetime.now(),
                    carbon_intensity=80.0,  # Lower intensity
                    unit=CarbonIntensityUnit.GRAMS_CO2_PER_KWH,
                    data_source="test"
                )
            }
            
            # Mock provider method
            mock_provider.find_cleanest_region.return_value = "EU-FR"
            
            cleanest = monitor.get_cleanest_region()
            assert cleanest == "EU-FR"
    
    def test_get_current_status(self, mock_provider):
        """Test getting current monitoring status."""
        with patch('carbon_aware_trainer.core.monitor.ElectricityMapProvider', return_value=mock_provider):
            monitor = CarbonMonitor(
                regions=["US-CA", "EU-FR"],
                data_source=CarbonDataSource.ELECTRICITYMAP
            )
            
            # Add some test data
            test_intensity = CarbonIntensity(
                region="US-CA",
                timestamp=datetime.now(),
                carbon_intensity=120.0,
                unit=CarbonIntensityUnit.GRAMS_CO2_PER_KWH,
                data_source="test",
                renewable_percentage=45.0
            )
            monitor._current_intensities["US-CA"] = test_intensity
            
            status = monitor.get_current_status()
            
            assert status['regions'] == ["US-CA", "EU-FR"]
            assert status['data_source'] == 'electricitymap'
            assert status['monitoring_active'] is False  # Not started
            assert 'current_intensities' in status
            assert 'US-CA' in status['current_intensities']
            assert status['current_intensities']['US-CA']['carbon_intensity'] == 120.0