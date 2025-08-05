"""Main carbon-aware training scheduler."""

import time
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
import logging
from contextlib import asynccontextmanager

from ..core.types import (
    TrainingState, TrainingConfig, TrainingMetrics, 
    CarbonIntensity, OptimalWindow
)
from ..core.monitor import CarbonMonitor


logger = logging.getLogger(__name__)


class CarbonAwareTrainer:
    """Main carbon-aware training scheduler that wraps ML training loops."""
    
    def __init__(
        self,
        model: Any = None,
        optimizer: Any = None,
        carbon_model: str = 'electricitymap',
        region: str = 'US-CA',
        target_carbon_intensity: float = 100.0,
        config: Optional[TrainingConfig] = None,
        api_key: Optional[str] = None
    ):
        """Initialize carbon-aware trainer.
        
        Args:
            model: ML model to train (PyTorch, TensorFlow, etc.)
            optimizer: Model optimizer
            carbon_model: Carbon data source ('electricitymap', 'watttime', 'cached')
            region: Training region code
            target_carbon_intensity: Target carbon intensity threshold (gCO2/kWh)
            config: Advanced training configuration
            api_key: API key for carbon data source
        """
        self.model = model
        self.optimizer = optimizer
        self.region = region
        self.api_key = api_key
        
        # Training configuration
        if config:
            self.config = config
        else:
            self.config = TrainingConfig(
                carbon_threshold=target_carbon_intensity,
                pause_threshold=target_carbon_intensity * 1.5,
                resume_threshold=target_carbon_intensity * 0.8
            )
        
        # Carbon monitoring
        self.monitor: Optional[CarbonMonitor] = None
        
        # Training state
        self.state = TrainingState.STOPPED
        self.session_id = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.step = 0
        self.epoch = 0
        
        # Metrics tracking
        self.metrics = TrainingMetrics(
            session_id=self.session_id,
            start_time=datetime.now()
        )
        
        # Callbacks
        self._state_callbacks: List[Callable] = []
        self._pause_start_time: Optional[datetime] = None
        
        # Training control
        self._should_pause = False
        self._force_stop = False
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
    
    async def initialize(self) -> None:
        """Initialize carbon monitoring and training infrastructure."""
        # Convert carbon_model string to enum
        from ..core.types import CarbonDataSource
        data_source_map = {
            'electricitymap': CarbonDataSource.ELECTRICITYMAP,
            'watttime': CarbonDataSource.WATTTIME,
            'cached': CarbonDataSource.CACHED
        }
        
        data_source = data_source_map.get(
            getattr(self, 'carbon_model', 'electricitymap').lower(),
            CarbonDataSource.ELECTRICITYMAP
        )
        
        # Initialize carbon monitor
        self.monitor = CarbonMonitor(
            regions=[self.region] + self.config.preferred_regions,
            data_source=data_source,
            api_key=self.api_key,
            update_interval=self.config.check_interval
        )
        
        await self.monitor.__aenter__()
        
        # Add carbon intensity callback
        self.monitor.add_callback(self._on_carbon_change)
        
        # Start monitoring
        await self.monitor.start_monitoring()
        
        logger.info(f"Initialized carbon-aware trainer for region {self.region}")
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.monitor:
            await self.monitor.__aexit__(None, None, None)
        
        # Finalize metrics
        self.metrics.end_time = datetime.now()
        
        logger.info(f"Training session {self.session_id} completed")
        logger.info(f"Total carbon emissions: {self.metrics.total_carbon_kg:.2f} kg CO2")
    
    async def _on_carbon_change(self, event_type: str, data: Dict[str, Any]) -> None:
        """Handle carbon intensity changes."""
        if event_type == 'intensity_change':
            region = data['region']
            new_intensity = data['new_intensity']
            
            if region == self.region:
                await self._evaluate_training_decision(new_intensity)
    
    async def _evaluate_training_decision(self, intensity: CarbonIntensity) -> None:
        """Evaluate whether to pause/resume training based on carbon intensity."""
        carbon_value = intensity.carbon_intensity
        
        # Update metrics
        if carbon_value > self.metrics.peak_carbon_intensity:
            self.metrics.peak_carbon_intensity = carbon_value
        if carbon_value < self.metrics.min_carbon_intensity:
            self.metrics.min_carbon_intensity = carbon_value
        
        # Make training decision
        if self.state == TrainingState.RUNNING:
            if carbon_value > self.config.pause_threshold:
                await self._pause_training(f"Carbon intensity {carbon_value:.1f} > threshold {self.config.pause_threshold:.1f}")
        
        elif self.state == TrainingState.PAUSED:
            if carbon_value < self.config.resume_threshold:
                await self._resume_training(f"Carbon intensity {carbon_value:.1f} < resume threshold {self.config.resume_threshold:.1f}")
    
    async def _pause_training(self, reason: str) -> None:
        """Pause training due to high carbon intensity."""
        if self.state != TrainingState.RUNNING:
            return
        
        self.state = TrainingState.PAUSED
        self._pause_start_time = datetime.now()
        self._should_pause = True
        
        logger.info(f"Pausing training: {reason}")
        await self._notify_state_change()
    
    async def _resume_training(self, reason: str) -> None:
        """Resume training due to low carbon intensity."""
        if self.state != TrainingState.PAUSED:
            return
        
        self.state = TrainingState.RUNNING
        
        # Update pause duration metrics
        if self._pause_start_time:
            pause_duration = datetime.now() - self._pause_start_time
            self.metrics.paused_duration += pause_duration
            self._pause_start_time = None
        
        self._should_pause = False
        
        logger.info(f"Resuming training: {reason}")
        await self._notify_state_change()
    
    async def _notify_state_change(self) -> None:
        """Notify callbacks of state changes."""
        for callback in self._state_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self.state, self.metrics)
                else:
                    callback(self.state, self.metrics)
            except Exception as e:
                logger.error(f"Error in state callback: {e}")
    
    def add_state_callback(self, callback: Callable) -> None:
        """Add training state change callback.
        
        Args:
            callback: Function (state, metrics) -> None
        """
        self._state_callbacks.append(callback)
    
    async def train_step(self, batch: Any, **kwargs) -> Any:
        """Execute a single training step with carbon awareness.
        
        Args:
            batch: Training batch data
            **kwargs: Additional arguments passed to training function
            
        Returns:
            Training step result (loss, metrics, etc.)
        """
        # Wait if training is paused
        await self._wait_for_resume()
        
        if self._force_stop:
            raise StopIteration("Training stopped by carbon-aware scheduler")
        
        # Update step counter
        self.step += 1
        
        # Calculate energy consumption (simplified)
        step_start_time = time.time()
        
        # Execute the actual training step
        result = await self._execute_training_step(batch, **kwargs)
        
        step_duration = time.time() - step_start_time
        
        # Estimate energy consumption (very simplified)
        # TODO: Integrate with actual GPU power monitoring
        estimated_power_kw = 0.4  # Assume 400W GPU power draw
        step_energy_kwh = (estimated_power_kw * step_duration) / 3600
        
        # Update metrics
        self.metrics.total_energy_kwh += step_energy_kwh
        
        # Get current carbon intensity for emissions calculation
        if self.monitor:
            current_intensity = await self.monitor.get_current_intensity(self.region)
            if current_intensity:
                step_carbon_kg = step_energy_kwh * (current_intensity.carbon_intensity / 1000)
                self.metrics.total_carbon_kg += step_carbon_kg
                
                # Update running average
                total_steps = self.step
                self.metrics.avg_carbon_intensity = (
                    (self.metrics.avg_carbon_intensity * (total_steps - 1) + 
                     current_intensity.carbon_intensity) / total_steps
                )
        
        return result
    
    async def _wait_for_resume(self) -> None:
        """Wait while training is paused."""
        while self._should_pause and not self._force_stop:
            await asyncio.sleep(1)
    
    async def _execute_training_step(self, batch: Any, **kwargs) -> Any:
        """Execute the actual training step.
        
        This is a placeholder - subclasses or integrations should override.
        """
        if hasattr(self.model, 'train_step'):
            return self.model.train_step(batch, **kwargs)
        elif hasattr(self.model, '__call__'):
            # Basic forward pass
            return self.model(batch)
        else:
            raise NotImplementedError(
                "No training step implementation found. "
                "Use framework-specific integrations or override _execute_training_step."
            )
    
    async def find_optimal_training_window(
        self,
        duration_hours: int,
        start_time: Optional[datetime] = None
    ) -> Optional[OptimalWindow]:
        """Find optimal training window based on carbon forecasts.
        
        Args:
            duration_hours: Required training duration
            start_time: Earliest start time (defaults to now)
            
        Returns:
            Optimal training window or None
        """
        if not self.monitor:
            raise RuntimeError("Monitor not initialized")
        
        return self.monitor.find_optimal_window(
            duration_hours=duration_hours,
            max_carbon_intensity=self.config.carbon_threshold,
            preferred_regions=[self.region] + self.config.preferred_regions
        )
    
    def get_carbon_metrics(self) -> Dict[str, Any]:
        """Get current carbon and training metrics.
        
        Returns:
            Dictionary with current metrics
        """
        runtime = datetime.now() - self.metrics.start_time
        
        return {
            'session_id': self.session_id,
            'current_state': self.state.value,
            'step': self.step,
            'epoch': self.epoch,
            'runtime_hours': runtime.total_seconds() / 3600,
            'total_energy_kwh': self.metrics.total_energy_kwh,
            'total_carbon_kg': self.metrics.total_carbon_kg,
            'avg_carbon_intensity': self.metrics.avg_carbon_intensity,
            'peak_carbon_intensity': self.metrics.peak_carbon_intensity,
            'min_carbon_intensity': self.metrics.min_carbon_intensity,
            'paused_duration_hours': self.metrics.paused_duration.total_seconds() / 3600,
            'carbon_saved_kg': self.metrics.carbon_saved_kg,
            'current_intensity': None
        }
    
    async def start_training(self) -> None:
        """Start carbon-aware training session."""
        if self.state != TrainingState.STOPPED:
            logger.warning("Training already started")
            return
        
        self.state = TrainingState.RUNNING
        self.metrics.start_time = datetime.now()
        
        logger.info(f"Started carbon-aware training session {self.session_id}")
        await self._notify_state_change()
    
    async def stop_training(self) -> None:
        """Stop training session."""
        self._force_stop = True
        self.state = TrainingState.STOPPED
        
        logger.info(f"Stopped training session {self.session_id}")
        await self._notify_state_change()
    
    @asynccontextmanager
    async def training_session(self):
        """Context manager for carbon-aware training session."""
        await self.start_training()
        try:
            yield self
        finally:
            await self.stop_training()