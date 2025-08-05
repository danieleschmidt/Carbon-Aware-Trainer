"""Unit tests for carbon-aware scheduler."""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from carbon_aware_trainer.core.scheduler import CarbonAwareTrainer
from carbon_aware_trainer.core.types import (
    TrainingState, TrainingConfig, CarbonIntensity, 
    CarbonIntensityUnit, CarbonDataSource
)


class TestCarbonAwareTrainer:
    """Test cases for CarbonAwareTrainer."""
    
    @pytest.fixture
    def basic_config(self):
        """Basic training configuration for tests."""
        return TrainingConfig(
            carbon_threshold=100.0,
            pause_threshold=150.0,
            resume_threshold=80.0,
            check_interval=60
        )
    
    @pytest.fixture
    def mock_monitor(self):
        """Mock carbon monitor."""
        monitor = AsyncMock()
        monitor.__aenter__ = AsyncMock(return_value=monitor)
        monitor.__aexit__ = AsyncMock(return_value=None)
        monitor.start_monitoring = AsyncMock()
        monitor.add_callback = MagicMock()
        return monitor
    
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
    
    def test_trainer_initialization(self, basic_config):
        """Test trainer initialization."""
        trainer = CarbonAwareTrainer(
            region="US-CA",
            target_carbon_intensity=100.0,
            config=basic_config
        )
        
        assert trainer.region == "US-CA"
        assert trainer.config.carbon_threshold == 100.0
        assert trainer.state == TrainingState.STOPPED
        assert trainer.step == 0
        assert trainer.epoch == 0
    
    @pytest.mark.asyncio
    async def test_trainer_context_manager(self, basic_config, mock_monitor):
        """Test trainer as async context manager."""
        with patch('carbon_aware_trainer.core.scheduler.CarbonMonitor', return_value=mock_monitor):
            trainer = CarbonAwareTrainer(
                region="US-CA",
                config=basic_config
            )
            
            async with trainer:
                assert trainer.monitor is not None
                mock_monitor.__aenter__.assert_called_once()
                mock_monitor.start_monitoring.assert_called_once()
            
            mock_monitor.__aexit__.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_training_session(self, basic_config, mock_monitor):
        """Test training session lifecycle."""
        with patch('carbon_aware_trainer.core.scheduler.CarbonMonitor', return_value=mock_monitor):
            trainer = CarbonAwareTrainer(
                region="US-CA",
                config=basic_config
            )
            
            async with trainer:
                # Test starting training
                async with trainer.training_session():
                    assert trainer.state == TrainingState.RUNNING
                
                # Training should be stopped after context exit
                assert trainer.state == TrainingState.STOPPED
    
    @pytest.mark.asyncio
    async def test_carbon_intensity_evaluation(self, basic_config, mock_monitor, sample_intensity):
        """Test carbon intensity evaluation and decision making."""
        with patch('carbon_aware_trainer.core.scheduler.CarbonMonitor', return_value=mock_monitor):
            trainer = CarbonAwareTrainer(
                region="US-CA",
                config=basic_config
            )
            
            async with trainer:
                await trainer.start_training()
                
                # Test pausing on high carbon intensity
                high_intensity = CarbonIntensity(
                    region="US-CA",
                    timestamp=datetime.now(),
                    carbon_intensity=200.0,  # Above pause threshold (150)
                    unit=CarbonIntensityUnit.GRAMS_CO2_PER_KWH,
                    data_source="test"
                )
                
                await trainer._evaluate_training_decision(high_intensity)
                assert trainer.state == TrainingState.PAUSED
                assert trainer._should_pause is True
                
                # Test resuming on low carbon intensity
                low_intensity = CarbonIntensity(
                    region="US-CA",
                    timestamp=datetime.now(),
                    carbon_intensity=60.0,  # Below resume threshold (80)
                    unit=CarbonIntensityUnit.GRAMS_CO2_PER_KWH,
                    data_source="test"
                )
                
                await trainer._evaluate_training_decision(low_intensity)
                assert trainer.state == TrainingState.RUNNING
                assert trainer._should_pause is False
    
    @pytest.mark.asyncio
    async def test_training_step_metrics(self, basic_config, mock_monitor, sample_intensity):
        """Test training step execution and metrics tracking."""
        mock_monitor.get_current_intensity.return_value = sample_intensity
        
        with patch('carbon_aware_trainer.core.scheduler.CarbonMonitor', return_value=mock_monitor):
            trainer = CarbonAwareTrainer(
                region="US-CA",
                config=basic_config
            )
            
            # Mock model with simple training step
            mock_model = MagicMock()
            mock_model.train_step = MagicMock(return_value={"loss": 0.5})
            trainer.model = mock_model
            
            async with trainer:
                await trainer.start_training()
                
                # Execute training step
                batch_data = {"input": "test_data"}
                result = await trainer.train_step(batch_data)
                
                # Verify step execution
                assert result == {"loss": 0.5}
                assert trainer.step == 1
                
                # Verify metrics updated
                assert trainer.metrics.total_energy_kwh > 0
                assert trainer.metrics.total_carbon_kg > 0
                mock_monitor.get_current_intensity.assert_called_with("US-CA")
    
    @pytest.mark.asyncio
    async def test_pause_resume_flow(self, basic_config, mock_monitor):
        """Test training pause and resume flow."""
        with patch('carbon_aware_trainer.core.scheduler.CarbonMonitor', return_value=mock_monitor):
            trainer = CarbonAwareTrainer(
                region="US-CA",
                config=basic_config
            )
            
            state_changes = []
            
            def track_state_change(state, metrics):
                state_changes.append(state)
            
            async with trainer:
                trainer.add_state_callback(track_state_change)
                await trainer.start_training()
                
                # Pause training
                await trainer._pause_training("Test pause")
                assert trainer.state == TrainingState.PAUSED
                assert trainer._pause_start_time is not None
                
                # Wait a bit and resume
                await asyncio.sleep(0.1)
                await trainer._resume_training("Test resume")
                assert trainer.state == TrainingState.RUNNING
                assert trainer._pause_start_time is None
                assert trainer.metrics.paused_duration.total_seconds() > 0
                
                # Verify state change callbacks were called
                assert TrainingState.RUNNING in state_changes
                assert TrainingState.PAUSED in state_changes
    
    def test_get_carbon_metrics(self, basic_config, mock_monitor):
        """Test carbon metrics retrieval."""
        with patch('carbon_aware_trainer.core.scheduler.CarbonMonitor', return_value=mock_monitor):
            trainer = CarbonAwareTrainer(
                region="US-CA",
                config=basic_config
            )
            
            # Update some metrics
            trainer.step = 100
            trainer.epoch = 5
            trainer.metrics.total_energy_kwh = 2.5
            trainer.metrics.total_carbon_kg = 0.3
            trainer.metrics.avg_carbon_intensity = 120.0
            
            metrics = trainer.get_carbon_metrics()
            
            assert metrics['step'] == 100
            assert metrics['epoch'] == 5
            assert metrics['total_energy_kwh'] == 2.5
            assert metrics['total_carbon_kg'] == 0.3
            assert metrics['avg_carbon_intensity'] == 120.0
            assert 'session_id' in metrics
            assert 'current_state' in metrics
    
    @pytest.mark.asyncio
    async def test_optimal_training_window_search(self, basic_config, mock_monitor):
        """Test optimal training window finding."""
        from carbon_aware_trainer.core.types import OptimalWindow
        
        # Mock optimal window
        mock_window = OptimalWindow(
            start_time=datetime.now() + timedelta(hours=2),
            end_time=datetime.now() + timedelta(hours=10),
            avg_carbon_intensity=80.0,
            total_expected_carbon_kg=1.2,
            confidence_score=0.85,
            renewable_percentage=65.0,
            region="US-CA"
        )
        
        mock_monitor.find_optimal_window.return_value = mock_window
        
        with patch('carbon_aware_trainer.core.scheduler.CarbonMonitor', return_value=mock_monitor):
            trainer = CarbonAwareTrainer(
                region="US-CA",
                config=basic_config
            )
            
            async with trainer:
                window = await trainer.find_optimal_training_window(
                    duration_hours=8
                )
                
                assert window is not None
                assert window.avg_carbon_intensity == 80.0
                assert window.confidence_score == 0.85
                mock_monitor.find_optimal_window.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_force_stop_training(self, basic_config, mock_monitor):
        """Test force stopping training."""
        with patch('carbon_aware_trainer.core.scheduler.CarbonMonitor', return_value=mock_monitor):
            trainer = CarbonAwareTrainer(
                region="US-CA",
                config=basic_config
            )
            
            async with trainer:
                await trainer.start_training()
                await trainer.stop_training()
                
                # Training step should raise StopIteration
                with pytest.raises(StopIteration, match="Training stopped"):
                    await trainer.train_step({"data": "test"})