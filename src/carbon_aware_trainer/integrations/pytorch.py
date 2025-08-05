"""PyTorch integration for carbon-aware training."""

import time
import logging
from typing import Any, Dict, Optional, Union, Callable
from datetime import datetime, timedelta

from ..core.scheduler import CarbonAwareTrainer
from ..core.types import TrainingConfig, TrainingState
from ..monitoring.metrics import MetricsCollector, TrainingStep


logger = logging.getLogger(__name__)


class CarbonAwarePyTorchTrainer(CarbonAwareTrainer):
    """PyTorch-specific carbon-aware trainer with deep integration."""
    
    def __init__(
        self,
        model: Any,
        optimizer: Any,
        loss_function: Optional[Callable] = None,
        region: str = 'US-CA',
        carbon_model: str = 'electricitymap',
        config: Optional[TrainingConfig] = None,
        api_key: Optional[str] = None,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """Initialize PyTorch carbon-aware trainer.
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            loss_function: Loss function (if None, assumes model returns loss)
            region: Training region
            carbon_model: Carbon data source
            config: Training configuration
            api_key: API key for carbon data
            metrics_collector: Custom metrics collector
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            carbon_model=carbon_model,
            region=region,
            config=config,
            api_key=api_key
        )
        
        self.loss_function = loss_function
        self.device = self._detect_device()
        
        # Initialize metrics collection
        self.metrics_collector = metrics_collector or MetricsCollector(
            session_id=self.session_id
        )
        
        # PyTorch-specific state
        self._scaler = None
        self._mixed_precision = False
        self._compile_model = False
        
        # Performance tracking
        self._step_times = []
        self._memory_usage = []
    
    def _detect_device(self) -> str:
        """Detect PyTorch device (CPU/GPU)."""
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                return f"cuda:{device}"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        except ImportError:
            logger.warning("PyTorch not available, assuming CPU")
            return "cpu"
    
    def enable_mixed_precision(self) -> None:
        """Enable automatic mixed precision training."""
        try:
            import torch
            self._scaler = torch.cuda.amp.GradScaler()
            self._mixed_precision = True
            logger.info("Mixed precision training enabled")
        except (ImportError, AttributeError):
            logger.warning("Mixed precision not available")
    
    def enable_model_compilation(self) -> None:
        """Enable PyTorch 2.0 model compilation."""
        try:
            import torch
            if hasattr(torch, 'compile'):
                self.model = torch.compile(self.model)
                self._compile_model = True
                logger.info("Model compilation enabled")
            else:
                logger.warning("Model compilation not available (requires PyTorch 2.0+)")
        except ImportError:
            logger.warning("PyTorch not available for compilation")
    
    async def train_epoch(
        self,
        dataloader: Any,
        epoch: int,
        log_interval: int = 100
    ) -> Dict[str, float]:
        """Train for one epoch with carbon awareness.
        
        Args:
            dataloader: PyTorch DataLoader
            epoch: Current epoch number
            log_interval: Steps between logging
            
        Returns:
            Dictionary with epoch metrics
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for this trainer")
        
        self.model.train()
        self.epoch = epoch
        
        # Start epoch tracking
        self.metrics_collector.start_epoch(epoch)
        
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(dataloader):
            # Check if training should pause due to carbon intensity
            await self._wait_for_resume()
            
            if self._force_stop:
                break
            
            # Execute training step
            step_metrics = await self._pytorch_training_step(batch_data, batch_idx, epoch)
            
            epoch_loss += step_metrics.get('loss', 0.0)
            num_batches += 1
            
            # Log progress
            if batch_idx % log_interval == 0:
                current_metrics = self.get_carbon_metrics()
                logger.info(
                    f"Epoch {epoch}, Step {batch_idx}: "
                    f"Loss={step_metrics.get('loss', 0):.4f}, "
                    f"Carbon={current_metrics['total_carbon_kg']:.3f}kg"
                )
        
        # End epoch tracking
        avg_loss = epoch_loss / max(num_batches, 1)
        epoch_summary = self.metrics_collector.end_epoch(epoch)
        
        return {
            'epoch': epoch,
            'avg_loss': avg_loss,
            'num_batches': num_batches,
            'carbon_kg': epoch_summary.total_carbon_kg
        }
    
    async def _pytorch_training_step(
        self,
        batch_data: Any,
        batch_idx: int,
        epoch: int
    ) -> Dict[str, Any]:
        """Execute PyTorch training step with carbon awareness.
        
        Args:
            batch_data: Batch data from DataLoader
            batch_idx: Batch index
            epoch: Current epoch
            
        Returns:
            Dictionary with step metrics
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required")
        
        step_start_time = time.perf_counter()
        
        # Move data to device
        if isinstance(batch_data, (list, tuple)):
            inputs, targets = batch_data[0], batch_data[1]
        else:
            inputs, targets = batch_data, None
        
        if hasattr(inputs, 'to'):
            inputs = inputs.to(self.device)
        if targets is not None and hasattr(targets, 'to'):
            targets = targets.to(self.device)
        
        # Track memory usage
        if 'cuda' in self.device:
            try:
                memory_allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
                self._memory_usage.append(memory_allocated)
            except:
                memory_allocated = None
        else:
            memory_allocated = None
        
        # Forward pass with optional mixed precision
        self.optimizer.zero_grad()
        
        if self._mixed_precision and self._scaler:
            with torch.cuda.amp.autocast():
                if self.loss_function:
                    outputs = self.model(inputs)
                    loss = self.loss_function(outputs, targets)
                else:
                    # Assume model returns loss directly
                    loss = self.model(inputs)
            
            # Backward pass with gradient scaling
            self._scaler.scale(loss).backward()
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            # Standard precision training
            if self.loss_function:
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, targets)
            else:
                loss = self.model(inputs)
            
            loss.backward()
            self.optimizer.step()
        
        step_end_time = time.perf_counter()
        step_duration_ms = (step_end_time - step_start_time) * 1000
        
        # Update global step counter
        self.step += 1
        
        # Get current carbon intensity
        current_intensity = None
        if self.monitor:
            intensity_data = await self.monitor.get_current_intensity(self.region)
            if intensity_data:
                current_intensity = intensity_data.carbon_intensity
        
        # Estimate power consumption (simplified)
        gpu_power_watts = self._estimate_gpu_power()
        
        # Extract loss value
        loss_value = loss.item() if hasattr(loss, 'item') else float(loss)
        
        # Calculate additional metrics
        learning_rate = self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else None
        batch_size = inputs.shape[0] if hasattr(inputs, 'shape') else None
        
        # Log to metrics collector
        self.metrics_collector.log_training_step(
            step=self.step,
            loss=loss_value,
            learning_rate=learning_rate,
            batch_size=batch_size,
            power_watts=gpu_power_watts,
            carbon_intensity=current_intensity,
            duration_ms=step_duration_ms,
            memory_usage_mb=memory_allocated
        )
        
        # Update parent class metrics
        await self._update_parent_metrics(step_duration_ms, gpu_power_watts, current_intensity)
        
        return {
            'loss': loss_value,
            'duration_ms': step_duration_ms,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'memory_mb': memory_allocated,
            'power_watts': gpu_power_watts
        }
    
    def _estimate_gpu_power(self) -> Optional[float]:
        """Estimate GPU power consumption.
        
        Returns:
            Estimated power consumption in watts
        """
        if 'cuda' not in self.device:
            return None
        
        try:
            import torch
            
            # Get GPU utilization
            if torch.cuda.is_available():
                # Simplified power estimation based on memory usage
                memory_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                
                # Estimate based on typical GPU power consumption
                # This is a rough approximation - real implementation would use nvidia-ml-py
                base_power = 250  # Watts for typical training GPU
                estimated_power = base_power * (0.3 + 0.7 * memory_usage)  # 30% idle + 70% proportional
                
                return estimated_power
            
        except Exception as e:
            logger.debug(f"Could not estimate GPU power: {e}")
        
        return None
    
    async def _update_parent_metrics(
        self,
        step_duration_ms: float,
        power_watts: Optional[float],
        carbon_intensity: Optional[float]
    ) -> None:
        """Update parent class metrics.
        
        Args:
            step_duration_ms: Step duration in milliseconds
            power_watts: Power consumption in watts
            carbon_intensity: Carbon intensity in gCO2/kWh
        """
        # Estimate energy consumption
        if power_watts:
            step_energy_kwh = (power_watts * step_duration_ms / 1000) / 3600000
            self.metrics.total_energy_kwh += step_energy_kwh
            
            # Calculate carbon emissions
            if carbon_intensity:
                step_carbon_kg = step_energy_kwh * (carbon_intensity / 1000)
                self.metrics.total_carbon_kg += step_carbon_kg
                
                # Update running averages
                self.metrics.avg_carbon_intensity = (
                    (self.metrics.avg_carbon_intensity * (self.step - 1) + carbon_intensity) / self.step
                )
                
                # Update peak/min tracking
                if carbon_intensity > self.metrics.peak_carbon_intensity:
                    self.metrics.peak_carbon_intensity = carbon_intensity
                if carbon_intensity < self.metrics.min_carbon_intensity:
                    self.metrics.min_carbon_intensity = carbon_intensity
    
    async def validate_model(
        self,
        dataloader: Any,
        epoch: int
    ) -> Dict[str, float]:
        """Validate model with carbon tracking.
        
        Args:
            dataloader: Validation DataLoader
            epoch: Current epoch
            
        Returns:
            Validation metrics
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required")
        
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data in dataloader:
                # Move data to device
                if isinstance(batch_data, (list, tuple)):
                    inputs, targets = batch_data[0], batch_data[1]
                else:
                    inputs, targets = batch_data, None
                
                if hasattr(inputs, 'to'):
                    inputs = inputs.to(self.device)
                if targets is not None and hasattr(targets, 'to'):
                    targets = targets.to(self.device)
                
                # Forward pass
                if self.loss_function and targets is not None:
                    outputs = self.model(inputs)
                    loss = self.loss_function(outputs, targets)
                    val_loss += loss.item()
                    
                    # Calculate accuracy for classification
                    if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                        _, predicted = torch.max(outputs.data, 1)
                        total += targets.size(0)
                        correct += (predicted == targets).sum().item()
                else:
                    # Model returns loss directly
                    loss = self.model(inputs)
                    val_loss += loss.item()
        
        avg_val_loss = val_loss / len(dataloader)
        accuracy = correct / total if total > 0 else None
        
        logger.info(
            f"Validation Epoch {epoch}: "
            f"Loss={avg_val_loss:.4f}" +
            (f", Accuracy={accuracy:.4f}" if accuracy is not None else "")
        )
        
        return {
            'val_loss': avg_val_loss,
            'val_accuracy': accuracy,
            'epoch': epoch
        }
    
    def save_checkpoint(
        self,
        filepath: str,
        epoch: int,
        additional_state: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Save training checkpoint with carbon metrics.
        
        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch
            additional_state: Additional state to save
            
        Returns:
            True if checkpoint saved successfully
        """
        try:
            import torch
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'carbon_metrics': self.get_carbon_metrics(),
                'session_summary': self.metrics_collector.get_session_summary(),
                'training_config': {
                    'carbon_threshold': self.config.carbon_threshold,
                    'pause_threshold': self.config.pause_threshold,
                    'resume_threshold': self.config.resume_threshold,
                    'region': self.region
                }
            }
            
            if self._scaler:
                checkpoint['scaler_state_dict'] = self._scaler.state_dict()
            
            if additional_state:
                checkpoint.update(additional_state)
            
            torch.save(checkpoint, filepath)
            logger.info(f"Checkpoint saved: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False
    
    def load_checkpoint(
        self,
        filepath: str,
        load_optimizer: bool = True,
        load_scaler: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Load training checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
            load_scaler: Whether to load scaler state
            
        Returns:
            Checkpoint data or None if failed
        """
        try:
            import torch
            
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            if load_optimizer and 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scaler state
            if load_scaler and self._scaler and 'scaler_state_dict' in checkpoint:
                self._scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            # Restore epoch
            if 'epoch' in checkpoint:
                self.epoch = checkpoint['epoch']
            
            logger.info(f"Checkpoint loaded: {filepath}")
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get PyTorch-specific performance statistics.
        
        Returns:
            Performance statistics
        """
        stats = {
            'device': self.device,
            'mixed_precision': self._mixed_precision,
            'model_compiled': self._compile_model,
            'current_epoch': self.epoch,
            'total_steps': self.step
        }
        
        # Add step timing statistics
        if self._step_times:
            stats.update({
                'avg_step_time_ms': sum(self._step_times) / len(self._step_times),
                'min_step_time_ms': min(self._step_times),
                'max_step_time_ms': max(self._step_times)
            })
        
        # Add memory statistics
        if self._memory_usage:
            stats.update({
                'avg_memory_mb': sum(self._memory_usage) / len(self._memory_usage),
                'peak_memory_mb': max(self._memory_usage),
                'current_memory_mb': self._memory_usage[-1] if self._memory_usage else None
            })
        
        return stats