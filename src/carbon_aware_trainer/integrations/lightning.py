"""PyTorch Lightning integration for carbon-aware training."""

import logging
from typing import Any, Dict, Optional, Union
from datetime import datetime

from ..core.scheduler import CarbonAwareTrainer
from ..core.types import TrainingConfig, TrainingState
from ..monitoring.metrics import MetricsCollector


logger = logging.getLogger(__name__)


class CarbonAwareCallback:
    """PyTorch Lightning callback for carbon-aware training."""
    
    def __init__(
        self,
        pause_threshold: float = 150.0,
        resume_threshold: float = 80.0,
        carbon_model: str = 'electricitymap',
        region: str = 'US-CA',
        api_key: Optional[str] = None,
        migration_enabled: bool = False,
        regions: Optional[list] = None,
        check_interval: int = 300
    ):
        """Initialize carbon-aware callback.
        
        Args:
            pause_threshold: Carbon intensity to pause training (gCO2/kWh)
            resume_threshold: Carbon intensity to resume training (gCO2/kWh)
            carbon_model: Carbon data source
            region: Primary training region
            api_key: API key for carbon data
            migration_enabled: Enable cross-region migration
            regions: Available regions for migration
            check_interval: Carbon check interval in seconds
        """
        self.pause_threshold = pause_threshold
        self.resume_threshold = resume_threshold
        self.carbon_model = carbon_model
        self.region = region
        self.api_key = api_key
        self.migration_enabled = migration_enabled
        self.regions = regions or [region]
        self.check_interval = check_interval
        
        # Internal state
        self.carbon_trainer: Optional[CarbonAwareTrainer] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        self._is_paused = False
        self._last_carbon_check = datetime.now()
        
        # Lightning integration state
        self._trainer = None
        self._model = None
    
    def setup(self, trainer, pl_module, stage: str) -> None:
        """Setup callback when training starts.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: PyTorch Lightning module
            stage: Training stage
        """
        if stage != 'fit':
            return
        
        self._trainer = trainer
        self._model = pl_module
        
        # Initialize carbon-aware trainer
        config = TrainingConfig(
            carbon_threshold=(self.pause_threshold + self.resume_threshold) / 2,
            pause_threshold=self.pause_threshold,
            resume_threshold=self.resume_threshold,
            check_interval=self.check_interval,
            migration_enabled=self.migration_enabled,
            preferred_regions=self.regions
        )
        
        self.carbon_trainer = CarbonAwareTrainer(
            model=pl_module,
            optimizer=None,  # Lightning manages optimizer
            carbon_model=self.carbon_model,
            region=self.region,
            config=config,
            api_key=self.api_key
        )
        
        # Initialize metrics collector
        session_id = f"lightning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.metrics_collector = MetricsCollector(session_id=session_id)
        
        logger.info(f"Carbon-aware callback initialized for region {self.region}")
    
    async def on_train_start(self, trainer, pl_module) -> None:
        """Called when training starts.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: PyTorch Lightning module
        """
        if self.carbon_trainer:
            await self.carbon_trainer.initialize()
            await self.carbon_trainer.start_training()
    
    async def on_train_end(self, trainer, pl_module) -> None:
        """Called when training ends.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: PyTorch Lightning module
        """
        if self.carbon_trainer:
            await self.carbon_trainer.stop_training()
            await self.carbon_trainer.cleanup()
    
    async def on_train_epoch_start(self, trainer, pl_module) -> None:
        """Called at the start of each epoch.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: PyTorch Lightning module
        """
        if self.metrics_collector:
            self.metrics_collector.start_epoch(trainer.current_epoch)
    
    async def on_train_epoch_end(self, trainer, pl_module) -> None:
        """Called at the end of each epoch.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: PyTorch Lightning module
        """
        if self.metrics_collector:
            try:
                epoch_summary = self.metrics_collector.end_epoch(trainer.current_epoch)
                
                # Log epoch carbon metrics
                logger.info(
                    f"Epoch {trainer.current_epoch} carbon metrics: "
                    f"{epoch_summary.total_carbon_kg:.3f} kg CO2, "
                    f"avg intensity: {epoch_summary.avg_carbon_intensity:.1f} gCO2/kWh"
                )
                
                # Log to Lightning logger if available
                if trainer.logger:
                    trainer.logger.log_metrics({
                        'carbon/epoch_emissions_kg': epoch_summary.total_carbon_kg,
                        'carbon/avg_intensity': epoch_summary.avg_carbon_intensity,
                        'carbon/paused_duration_minutes': epoch_summary.paused_duration_seconds / 60
                    }, step=trainer.global_step)
                
            except Exception as e:
                logger.error(f"Error ending epoch carbon tracking: {e}")
    
    async def on_train_batch_start(
        self, 
        trainer, 
        pl_module, 
        batch: Any, 
        batch_idx: int
    ) -> Optional[int]:
        """Called before each training batch.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: PyTorch Lightning module
            batch: Training batch
            batch_idx: Batch index
            
        Returns:
            -1 to skip batch if carbon intensity too high
        """
        # Check if we should pause due to carbon intensity
        now = datetime.now()
        if (now - self._last_carbon_check).total_seconds() >= self.check_interval:
            should_pause = await self._check_carbon_intensity()
            self._last_carbon_check = now
            
            if should_pause and not self._is_paused:
                self._is_paused = True
                logger.warning("Pausing training due to high carbon intensity")
                
                # In Lightning, we can't directly pause, but we can skip batches
                return -1
            elif not should_pause and self._is_paused:
                self._is_paused = False
                logger.info("Resuming training - carbon intensity decreased")
        
        # Skip batch if currently paused
        if self._is_paused:
            return -1
        
        return None
    
    async def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch: Any,
        batch_idx: int
    ) -> None:
        """Called after each training batch.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: PyTorch Lightning module
            outputs: Training step outputs
            batch: Training batch
            batch_idx: Batch index
        """
        if not self.carbon_trainer or not self.metrics_collector:
            return
        
        try:
            # Extract metrics from Lightning outputs
            loss = None
            if isinstance(outputs, dict) and 'loss' in outputs:
                loss = outputs['loss'].item() if hasattr(outputs['loss'], 'item') else float(outputs['loss'])
            
            # Get current carbon intensity
            current_intensity = None
            if self.carbon_trainer.monitor:
                intensity_data = await self.carbon_trainer.monitor.get_current_intensity(self.region)
                if intensity_data:
                    current_intensity = intensity_data.carbon_intensity
            
            # Estimate power consumption
            power_watts = self._estimate_power_consumption()
            
            # Get batch size
            batch_size = None
            if hasattr(batch, '__len__'):
                batch_size = len(batch)
            elif isinstance(batch, (list, tuple)) and len(batch) > 0:
                if hasattr(batch[0], 'shape'):
                    batch_size = batch[0].shape[0]
            
            # Log training step
            self.metrics_collector.log_training_step(
                step=trainer.global_step,
                loss=loss,
                batch_size=batch_size,
                power_watts=power_watts,
                carbon_intensity=current_intensity
            )
            
            # Log to Lightning logger every 100 steps
            if trainer.global_step % 100 == 0 and trainer.logger:
                carbon_metrics = self.carbon_trainer.get_carbon_metrics()
                trainer.logger.log_metrics({
                    'carbon/total_emissions_kg': carbon_metrics['total_carbon_kg'],
                    'carbon/current_intensity': current_intensity or 0,
                    'carbon/paused_hours': carbon_metrics['paused_duration_hours']
                }, step=trainer.global_step)
                
        except Exception as e:
            logger.error(f"Error in carbon tracking batch end: {e}")
    
    async def _check_carbon_intensity(self) -> bool:
        """Check current carbon intensity and determine if should pause.
        
        Returns:
            True if training should be paused
        """
        if not self.carbon_trainer or not self.carbon_trainer.monitor:
            return False
        
        try:
            intensity_data = await self.carbon_trainer.monitor.get_current_intensity(self.region)
            if not intensity_data:
                return False
            
            carbon_intensity = intensity_data.carbon_intensity
            
            # Determine pause/resume based on thresholds
            if not self._is_paused and carbon_intensity > self.pause_threshold:
                return True
            elif self._is_paused and carbon_intensity < self.resume_threshold:
                return False
            
            return self._is_paused
            
        except Exception as e:
            logger.error(f"Error checking carbon intensity: {e}")
            return False
    
    def _estimate_power_consumption(self) -> Optional[float]:
        """Estimate current power consumption.
        
        Returns:
            Estimated power in watts
        """
        try:
            import torch
            
            if torch.cuda.is_available():
                # Simple estimation based on GPU memory usage
                memory_allocated = torch.cuda.memory_allocated()
                max_memory = torch.cuda.max_memory_allocated()
                
                if max_memory > 0:
                    memory_ratio = memory_allocated / max_memory
                    # Rough estimation: 300W base + proportional to memory usage
                    estimated_power = 300 * (0.3 + 0.7 * memory_ratio)
                    return estimated_power
            
        except Exception as e:
            logger.debug(f"Could not estimate power consumption: {e}")
        
        return None
    
    def on_save_checkpoint(
        self, 
        trainer, 
        pl_module, 
        checkpoint: Dict[str, Any]
    ) -> None:
        """Add carbon metrics to checkpoint.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: PyTorch Lightning module
            checkpoint: Checkpoint dictionary
        """
        if self.carbon_trainer and self.metrics_collector:
            try:
                checkpoint['carbon_metrics'] = {
                    'session_summary': self.metrics_collector.get_session_summary(),
                    'carbon_stats': self.carbon_trainer.get_carbon_metrics(),
                    'callback_config': {
                        'pause_threshold': self.pause_threshold,
                        'resume_threshold': self.resume_threshold,
                        'region': self.region,
                        'carbon_model': self.carbon_model
                    }
                }
            except Exception as e:
                logger.error(f"Error saving carbon metrics to checkpoint: {e}")
    
    def on_load_checkpoint(
        self, 
        trainer, 
        pl_module, 
        checkpoint: Dict[str, Any]
    ) -> None:
        """Restore carbon metrics from checkpoint.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: PyTorch Lightning module
            checkpoint: Checkpoint dictionary
        """
        if 'carbon_metrics' in checkpoint:
            try:
                carbon_data = checkpoint['carbon_metrics']
                logger.info("Restored carbon metrics from checkpoint")
                
                # Log restored metrics
                if 'carbon_stats' in carbon_data:
                    stats = carbon_data['carbon_stats']
                    logger.info(
                        f"Resumed with {stats.get('total_carbon_kg', 0):.3f} kg CO2 "
                        f"from {stats.get('total_steps', 0)} steps"
                    )
                    
            except Exception as e:
                logger.error(f"Error loading carbon metrics from checkpoint: {e}")
    
    def get_carbon_summary(self) -> Dict[str, Any]:
        """Get comprehensive carbon training summary.
        
        Returns:
            Dictionary with carbon metrics and statistics
        """
        if not self.carbon_trainer or not self.metrics_collector:
            return {"error": "Carbon trainer not initialized"}
        
        try:
            return {
                'session_summary': self.metrics_collector.get_session_summary(),
                'carbon_metrics': self.carbon_trainer.get_carbon_metrics(),
                'performance_stats': self.metrics_collector.get_performance_stats(),
                'carbon_stats': self.metrics_collector.get_carbon_stats(),
                'is_paused': self._is_paused,
                'region': self.region,
                'thresholds': {
                    'pause': self.pause_threshold,
                    'resume': self.resume_threshold
                }
            }
        except Exception as e:
            logger.error(f"Error getting carbon summary: {e}")
            return {"error": str(e)}


# Convenience factory function
def create_carbon_aware_callback(
    pause_threshold: float = 150.0,
    resume_threshold: float = 80.0,
    region: str = 'US-CA',
    **kwargs
) -> CarbonAwareCallback:
    """Create carbon-aware callback with sensible defaults.
    
    Args:
        pause_threshold: Carbon intensity to pause training
        resume_threshold: Carbon intensity to resume training
        region: Training region
        **kwargs: Additional callback arguments
        
    Returns:
        Configured CarbonAwareCallback
    """
    return CarbonAwareCallback(
        pause_threshold=pause_threshold,
        resume_threshold=resume_threshold,
        region=region,
        **kwargs
    )