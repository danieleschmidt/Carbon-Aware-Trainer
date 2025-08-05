"""Unit tests for framework integrations."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta

from carbon_aware_trainer.integrations.pytorch import CarbonAwarePyTorchTrainer
from carbon_aware_trainer.integrations.lightning import CarbonAwareCallback
from carbon_aware_trainer.core.types import TrainingConfig


class TestCarbonAwarePyTorchTrainer:
    """Test cases for PyTorch integration."""
    
    @pytest.fixture
    def mock_model(self):
        """Mock PyTorch model."""
        model = MagicMock()
        model.train = MagicMock()
        model.eval = MagicMock()
        model.state_dict = MagicMock(return_value={"layer1.weight": "mock_weights"})
        return model
    
    @pytest.fixture
    def mock_optimizer(self):
        """Mock PyTorch optimizer."""
        optimizer = MagicMock()
        optimizer.zero_grad = MagicMock()
        optimizer.step = MagicMock()
        optimizer.state_dict = MagicMock(return_value={"state": "mock_state"})
        optimizer.param_groups = [{"lr": 0.001}]
        return optimizer
    
    @pytest.fixture
    def mock_loss_function(self):
        """Mock loss function."""
        def loss_fn(outputs, targets):
            loss = MagicMock()
            loss.item = MagicMock(return_value=0.5)
            loss.backward = MagicMock()
            return loss
        return loss_fn
    
    @pytest.fixture
    def pytorch_trainer(self, mock_model, mock_optimizer, mock_loss_function):
        """Create PyTorch trainer for testing."""
        config = TrainingConfig(
            carbon_threshold=100.0,
            check_interval=60
        )
        
        with patch('carbon_aware_trainer.integrations.pytorch.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            
            trainer = CarbonAwarePyTorchTrainer(
                model=mock_model,
                optimizer=mock_optimizer,
                loss_function=mock_loss_function,
                region='US-CA',
                config=config
            )
            
            return trainer
    
    def test_device_detection(self, mock_model, mock_optimizer):
        """Test device detection."""
        with patch('carbon_aware_trainer.integrations.pytorch.torch') as mock_torch:
            # Test CUDA available
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.current_device.return_value = 0
            
            trainer = CarbonAwarePyTorchTrainer(
                model=mock_model,
                optimizer=mock_optimizer,
                region='US-CA'
            )
            
            assert trainer.device == "cuda:0"
    
    def test_mixed_precision_enable(self, pytorch_trainer):
        """Test enabling mixed precision."""
        with patch('carbon_aware_trainer.integrations.pytorch.torch') as mock_torch:
            mock_scaler = MagicMock()
            mock_torch.cuda.amp.GradScaler.return_value = mock_scaler
            
            pytorch_trainer.enable_mixed_precision()
            
            assert pytorch_trainer._mixed_precision is True
            assert pytorch_trainer._scaler == mock_scaler
    
    def test_model_compilation_enable(self, pytorch_trainer):
        """Test enabling model compilation."""
        with patch('carbon_aware_trainer.integrations.pytorch.torch') as mock_torch:
            mock_compile = MagicMock()
            mock_torch.compile = mock_compile
            mock_compile.return_value = "compiled_model"
            
            pytorch_trainer.enable_model_compilation()
            
            assert pytorch_trainer._compile_model is True
            assert pytorch_trainer.model == "compiled_model"
    
    @pytest.mark.asyncio
    async def test_pytorch_training_step(self, pytorch_trainer):
        """Test PyTorch training step execution."""
        # Mock tensor operations
        with patch('carbon_aware_trainer.integrations.pytorch.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            
            # Mock batch data
            mock_inputs = MagicMock()
            mock_inputs.shape = [32]  # Batch size of 32
            mock_targets = MagicMock()
            batch_data = (mock_inputs, mock_targets)
            
            # Mock model output
            mock_loss = MagicMock()
            mock_loss.item.return_value = 0.75
            pytorch_trainer.loss_function.return_value = mock_loss
            
            # Execute training step
            metrics = await pytorch_trainer._pytorch_training_step(batch_data, 0, 1)
            
            # Verify metrics
            assert metrics['loss'] == 0.75
            assert metrics['batch_size'] == 32
            assert 'duration_ms' in metrics
            
            # Verify optimizer was called
            pytorch_trainer.optimizer.zero_grad.assert_called_once()
            pytorch_trainer.optimizer.step.assert_called_once()
            mock_loss.backward.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_train_epoch(self, pytorch_trainer):
        """Test training epoch execution."""
        # Mock dataloader
        mock_dataloader = [
            (MagicMock(), MagicMock()),  # batch 1
            (MagicMock(), MagicMock()),  # batch 2
        ]
        
        # Mock batch processing
        with patch.object(pytorch_trainer, '_pytorch_training_step') as mock_step:
            mock_step.return_value = {'loss': 0.5}
            
            # Mock async context
            with patch.object(pytorch_trainer, '_wait_for_resume') as mock_wait:
                mock_wait.return_value = None
                
                epoch_metrics = await pytorch_trainer.train_epoch(mock_dataloader, epoch=1)
                
                assert epoch_metrics['epoch'] == 1
                assert epoch_metrics['avg_loss'] == 0.5
                assert epoch_metrics['num_batches'] == 2
                assert mock_step.call_count == 2
    
    @pytest.mark.asyncio
    async def test_validate_model(self, pytorch_trainer):
        """Test model validation."""
        with patch('carbon_aware_trainer.integrations.pytorch.torch') as mock_torch:
            mock_torch.no_grad.return_value.__enter__ = MagicMock()
            mock_torch.no_grad.return_value.__exit__ = MagicMock()
            
            # Mock validation data
            mock_outputs = MagicMock()
            mock_outputs.shape = [10, 5]  # 10 samples, 5 classes
            mock_outputs.data = MagicMock()
            
            mock_predicted = MagicMock()
            mock_torch.max.return_value = (MagicMock(), mock_predicted)
            
            mock_targets = MagicMock()
            mock_targets.size.return_value = 10
            
            mock_comparison = MagicMock()
            mock_comparison.sum.return_value.item.return_value = 8  # 8 correct out of 10
            mock_predicted.__eq__ = MagicMock(return_value=mock_comparison)
            
            pytorch_trainer.model.return_value = mock_outputs
            pytorch_trainer.loss_function.return_value.item.return_value = 0.3
            
            mock_dataloader = [(MagicMock(), mock_targets)]
            
            val_metrics = await pytorch_trainer.validate_model(mock_dataloader, epoch=1)
            
            assert val_metrics['epoch'] == 1
            assert val_metrics['val_loss'] == 0.3
            assert val_metrics['val_accuracy'] == 0.8  # 8/10
    
    def test_checkpoint_save_load(self, pytorch_trainer):
        """Test checkpoint saving and loading."""
        with patch('carbon_aware_trainer.integrations.pytorch.torch') as mock_torch:
            # Test saving
            mock_torch.save = MagicMock()
            
            success = pytorch_trainer.save_checkpoint("test_checkpoint.pth", epoch=5)
            
            assert success is True
            mock_torch.save.assert_called_once()
            
            # Verify checkpoint content
            saved_data = mock_torch.save.call_args[0][0]
            assert saved_data['epoch'] == 5
            assert 'model_state_dict' in saved_data
            assert 'optimizer_state_dict' in saved_data
            assert 'carbon_metrics' in saved_data
    
    def test_checkpoint_load(self, pytorch_trainer):
        """Test checkpoint loading."""
        with patch('carbon_aware_trainer.integrations.pytorch.torch') as mock_torch:
            # Mock checkpoint data
            mock_checkpoint = {
                'epoch': 10,
                'model_state_dict': {'layer1.weight': 'loaded_weights'},
                'optimizer_state_dict': {'state': 'loaded_state'},
                'carbon_metrics': {'total_carbon_kg': 1.5}
            }
            
            mock_torch.load.return_value = mock_checkpoint
            
            loaded_data = pytorch_trainer.load_checkpoint("test_checkpoint.pth")
            
            assert loaded_data is not None
            assert loaded_data['epoch'] == 10
            assert pytorch_trainer.epoch == 10
            
            # Verify model and optimizer state loading
            pytorch_trainer.model.load_state_dict.assert_called_once_with(
                mock_checkpoint['model_state_dict']
            )
            pytorch_trainer.optimizer.load_state_dict.assert_called_once_with(
                mock_checkpoint['optimizer_state_dict']
            )
    
    def test_performance_stats(self, pytorch_trainer):
        """Test performance statistics."""
        pytorch_trainer._step_times = [100, 150, 120]
        pytorch_trainer._memory_usage = [1024, 1536, 1200]
        
        stats = pytorch_trainer.get_performance_stats()
        
        assert stats['device'] == 'cpu'
        assert stats['mixed_precision'] is False
        assert stats['model_compiled'] is False
        assert stats['avg_step_time_ms'] == 123.33333333333333
        assert stats['min_step_time_ms'] == 100
        assert stats['max_step_time_ms'] == 150
        assert stats['avg_memory_mb'] == 1253.3333333333333
        assert stats['peak_memory_mb'] == 1536


class TestCarbonAwareCallback:
    """Test cases for Lightning callback."""
    
    @pytest.fixture
    def callback(self):
        """Create Lightning callback for testing."""
        return CarbonAwareCallback(
            pause_threshold=150.0,
            resume_threshold=80.0,
            region='US-CA',
            check_interval=300
        )
    
    @pytest.fixture
    def mock_trainer(self):
        """Mock Lightning trainer."""
        trainer = MagicMock()
        trainer.current_epoch = 1
        trainer.global_step = 100
        trainer.logger = MagicMock()
        return trainer
    
    @pytest.fixture
    def mock_pl_module(self):
        """Mock Lightning module."""
        module = MagicMock()
        return module
    
    def test_callback_initialization(self, callback):
        """Test callback initialization."""
        assert callback.pause_threshold == 150.0
        assert callback.resume_threshold == 80.0
        assert callback.region == 'US-CA'
        assert callback.check_interval == 300
        assert callback._is_paused is False
    
    def test_setup(self, callback, mock_trainer, mock_pl_module):
        """Test callback setup."""
        with patch('carbon_aware_trainer.integrations.lightning.CarbonAwareTrainer') as mock_trainer_class:
            mock_carbon_trainer = MagicMock()
            mock_trainer_class.return_value = mock_carbon_trainer
            
            callback.setup(mock_trainer, mock_pl_module, 'fit')
            
            assert callback.carbon_trainer == mock_carbon_trainer
            assert callback.metrics_collector is not None
            assert callback._trainer == mock_trainer
            assert callback._model == mock_pl_module
    
    @pytest.mark.asyncio
    async def test_train_start_end(self, callback, mock_trainer, mock_pl_module):
        """Test training start and end callbacks."""
        # Mock carbon trainer
        mock_carbon_trainer = AsyncMock()
        callback.carbon_trainer = mock_carbon_trainer
        
        # Test train start
        await callback.on_train_start(mock_trainer, mock_pl_module)
        mock_carbon_trainer.initialize.assert_called_once()
        mock_carbon_trainer.start_training.assert_called_once()
        
        # Test train end
        await callback.on_train_end(mock_trainer, mock_pl_module)
        mock_carbon_trainer.stop_training.assert_called_once()
        mock_carbon_trainer.cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_epoch_start_end(self, callback, mock_trainer, mock_pl_module):
        """Test epoch start and end callbacks."""
        # Mock metrics collector
        mock_metrics = MagicMock()
        callback.metrics_collector = mock_metrics
        
        # Test epoch start
        await callback.on_train_epoch_start(mock_trainer, mock_pl_module)
        mock_metrics.start_epoch.assert_called_once_with(1)
        
        # Test epoch end
        mock_epoch_summary = MagicMock()
        mock_epoch_summary.total_carbon_kg = 0.5
        mock_epoch_summary.avg_carbon_intensity = 120.0
        mock_epoch_summary.paused_duration_seconds = 300
        mock_metrics.end_epoch.return_value = mock_epoch_summary
        
        await callback.on_train_epoch_end(mock_trainer, mock_pl_module)
        mock_metrics.end_epoch.assert_called_once_with(1)
    
    @pytest.mark.asyncio
    async def test_batch_start_carbon_check(self, callback, mock_trainer, mock_pl_module):
        """Test batch start with carbon intensity check."""
        # Mock carbon trainer and monitor
        mock_carbon_trainer = AsyncMock()
        mock_monitor = AsyncMock()
        mock_carbon_trainer.monitor = mock_monitor
        callback.carbon_trainer = mock_carbon_trainer
        
        # Mock high carbon intensity
        from carbon_aware_trainer.core.types import CarbonIntensity, CarbonIntensityUnit
        high_intensity = CarbonIntensity(
            region="US-CA",
            timestamp=datetime.now(),
            carbon_intensity=200.0,  # Above pause threshold
            unit=CarbonIntensityUnit.GRAMS_CO2_PER_KWH,
            data_source="mock"
        )
        mock_monitor.get_current_intensity.return_value = high_intensity
        
        # Force carbon check by setting old timestamp
        callback._last_carbon_check = datetime.now() - timedelta(seconds=400)
        
        result = await callback.on_train_batch_start(mock_trainer, mock_pl_module, None, 0)
        
        # Should skip batch due to high carbon
        assert result == -1
        assert callback._is_paused is True
    
    @pytest.mark.asyncio
    async def test_batch_end_metrics_logging(self, callback, mock_trainer, mock_pl_module):
        """Test batch end metrics logging."""
        # Setup mocks
        mock_carbon_trainer = AsyncMock()
        mock_monitor = AsyncMock()
        mock_carbon_trainer.monitor = mock_monitor
        mock_metrics = MagicMock()
        
        callback.carbon_trainer = mock_carbon_trainer
        callback.metrics_collector = mock_metrics
        
        # Mock carbon intensity
        from carbon_aware_trainer.core.types import CarbonIntensity, CarbonIntensityUnit
        intensity = CarbonIntensity(
            region="US-CA",
            timestamp=datetime.now(),
            carbon_intensity=90.0,
            unit=CarbonIntensityUnit.GRAMS_CO2_PER_KWH,
            data_source="mock"
        )
        mock_monitor.get_current_intensity.return_value = intensity
        
        # Mock training outputs
        mock_loss = MagicMock()
        mock_loss.item.return_value = 0.3
        outputs = {'loss': mock_loss}
        
        # Mock batch with length
        batch = [1, 2, 3, 4]  # Batch size of 4
        
        await callback.on_batch_end(mock_trainer, mock_pl_module, outputs, batch, 0)
        
        # Verify metrics logging
        mock_metrics.log_training_step.assert_called_once()
        call_args = mock_metrics.log_training_step.call_args
        assert call_args[1]['step'] == 100
        assert call_args[1]['loss'] == 0.3
        assert call_args[1]['batch_size'] == 4
        assert call_args[1]['carbon_intensity'] == 90.0
    
    def test_checkpoint_save_load(self, callback, mock_trainer, mock_pl_module):
        """Test checkpoint integration."""
        # Setup mocks
        mock_carbon_trainer = MagicMock()
        mock_metrics = MagicMock()
        mock_session_summary = MagicMock()
        mock_carbon_stats = {'total_carbon_kg': 2.5}
        
        mock_carbon_trainer.get_carbon_metrics.return_value = mock_carbon_stats
        mock_metrics.get_session_summary.return_value = mock_session_summary
        
        callback.carbon_trainer = mock_carbon_trainer
        callback.metrics_collector = mock_metrics
        
        # Test saving to checkpoint
        checkpoint = {}
        callback.on_save_checkpoint(mock_trainer, mock_pl_module, checkpoint)
        
        assert 'carbon_metrics' in checkpoint
        assert 'session_summary' in checkpoint['carbon_metrics']
        assert 'carbon_stats' in checkpoint['carbon_metrics']
        assert 'callback_config' in checkpoint['carbon_metrics']
        
        # Test loading from checkpoint
        callback.on_load_checkpoint(mock_trainer, mock_pl_module, checkpoint)
        # Should complete without error
    
    def test_carbon_summary(self, callback):
        """Test carbon summary generation."""
        # Test without initialization
        summary = callback.get_carbon_summary()
        assert 'error' in summary
        
        # Test with initialization
        mock_carbon_trainer = MagicMock()
        mock_metrics = MagicMock()
        
        mock_carbon_trainer.get_carbon_metrics.return_value = {'total_carbon_kg': 1.0}
        mock_metrics.get_session_summary.return_value = MagicMock()
        mock_metrics.get_performance_stats.return_value = {'avg_step_time_ms': 100}
        mock_metrics.get_carbon_stats.return_value = {'avg_carbon_intensity': 95.0}
        
        callback.carbon_trainer = mock_carbon_trainer
        callback.metrics_collector = mock_metrics
        
        summary = callback.get_carbon_summary()
        
        assert 'session_summary' in summary
        assert 'carbon_metrics' in summary
        assert 'performance_stats' in summary
        assert 'carbon_stats' in summary
        assert summary['region'] == 'US-CA'
        assert summary['thresholds']['pause'] == 150.0
        assert summary['thresholds']['resume'] == 80.0
    
    def test_create_callback_factory(self):
        """Test convenience factory function."""
        from carbon_aware_trainer.integrations.lightning import create_carbon_aware_callback
        
        callback = create_carbon_aware_callback(
            pause_threshold=200.0,
            resume_threshold=60.0,
            region='EU-FR'
        )
        
        assert isinstance(callback, CarbonAwareCallback)
        assert callback.pause_threshold == 200.0
        assert callback.resume_threshold == 60.0
        assert callback.region == 'EU-FR'