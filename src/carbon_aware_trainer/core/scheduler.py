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

try:
    from ..core.monitor import CarbonMonitor
except ImportError:
    CarbonMonitor = None

try:
    from .resilience_manager import ResilienceManager, FailureType
    HAS_RESILIENCE = True
except ImportError:
    ResilienceManager = None
    FailureType = None
    HAS_RESILIENCE = False

try:
    from .generation3_optimizer import Generation3Optimizer, OptimizationMode
    HAS_GENERATION3 = True
except ImportError:
    Generation3Optimizer = None
    OptimizationMode = None
    HAS_GENERATION3 = False


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
        self._carbon_model_str = carbon_model
        
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
        
        # Resilience management (Generation 2 feature)
        self.resilience_manager: Optional[ResilienceManager] = None
        self.enable_resilience = HAS_RESILIENCE and config and getattr(config, 'enable_resilience', True)
        
        # Performance optimization (Generation 3 feature) 
        self.performance_optimizer: Optional[Generation3Optimizer] = None
        self.enable_performance_optimization = HAS_GENERATION3 and config and getattr(config, 'resource_optimization_enabled', True)
        
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
        """Initialize carbon monitoring, resilience management and training infrastructure."""
        # Initialize resilience manager first (Generation 2)
        if self.enable_resilience and ResilienceManager:
            checkpoint_dir = f"./checkpoints/{self.session_id}"
            self.resilience_manager = ResilienceManager(
                checkpoint_dir=checkpoint_dir,
                max_failures=self.config.max_failures if hasattr(self.config, 'max_failures') else 5,
                checkpoint_interval=self.config.checkpoint_interval if hasattr(self.config, 'checkpoint_interval') else 100
            )
            await self.resilience_manager.initialize()
            
            # Add failure callback
            self.resilience_manager.add_failure_callback(self._on_training_failure)
            logger.info("Resilience manager initialized")
        
        # Initialize performance optimizer (Generation 3)
        if self.enable_performance_optimization and Generation3Optimizer and OptimizationMode:
            optimization_mode = getattr(self.config, 'optimization_mode', 'balanced')
            if isinstance(optimization_mode, str):
                mode_map = {
                    'carbon_first': OptimizationMode.CARBON_FIRST,
                    'performance_first': OptimizationMode.PERFORMANCE_FIRST,
                    'balanced': OptimizationMode.BALANCED,
                    'cost_optimized': OptimizationMode.COST_OPTIMIZED
                }
                optimization_mode = mode_map.get(optimization_mode, OptimizationMode.BALANCED)
            
            self.performance_optimizer = Generation3Optimizer(
                optimization_mode=optimization_mode,
                target_utilization=getattr(self.config, 'target_utilization', 0.8),
                max_workers=getattr(self.config, 'max_concurrent_jobs', 4)
            )
            await self.performance_optimizer.initialize()
            logger.info("Performance optimizer initialized")
        
        # Convert carbon_model string to enum
        from ..core.types import CarbonDataSource
        data_source_map = {
            'electricitymap': CarbonDataSource.ELECTRICITYMAP,
            'watttime': CarbonDataSource.WATTTIME,
            'cached': CarbonDataSource.CACHED
        }
        
        # Fix: Use self.carbon_model if it exists, otherwise fall back to 'electricitymap'
        carbon_model_str = getattr(self, 'carbon_model', 'electricitymap')
        if hasattr(self, '_carbon_model_str'):
            carbon_model_str = self._carbon_model_str
        
        data_source = data_source_map.get(
            carbon_model_str.lower(),
            CarbonDataSource.ELECTRICITYMAP
        )
        
        # Initialize carbon monitor with resilience protection
        if CarbonMonitor:
            try:
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
                
            except Exception as e:
                logger.error(f"Failed to initialize carbon monitor: {e}")
                if self.resilience_manager and FailureType:
                    await self.resilience_manager.handle_training_failure(
                        FailureType.CARBON_API, e, {'initialization': True}
                    )
        
        logger.info(f"Initialized carbon-aware trainer for region {self.region}")
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.monitor:
            await self.monitor.__aexit__(None, None, None)
        
        # Clean up resilience manager
        if self.resilience_manager:
            await self.resilience_manager.cleanup()
        
        # Clean up performance optimizer (Generation 3)
        if self.performance_optimizer:
            await self.performance_optimizer.cleanup()
        
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
        """Execute a single training step with carbon awareness and resilience.
        
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
        
        try:
            # Calculate energy consumption (simplified)
            step_start_time = time.time()
            
            # Execute the actual training step with resilience protection
            result = await self._execute_training_step_with_resilience(batch, **kwargs)
            
            step_duration = time.time() - step_start_time
            
            # Estimate energy consumption (very simplified)
            # TODO: Integrate with actual GPU power monitoring
            estimated_power_kw = 0.4  # Assume 400W GPU power draw
            step_energy_kwh = (estimated_power_kw * step_duration) / 3600
            
            # Update metrics
            self.metrics.total_energy_kwh += step_energy_kwh
            
            # Get current carbon intensity for emissions calculation
            if self.monitor:
                try:
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
                except Exception as e:
                    logger.warning(f"Failed to get carbon intensity: {e}")
                    if self.resilience_manager and FailureType:
                        await self.resilience_manager.handle_training_failure(
                            FailureType.CARBON_API, e, {'step': self.step}
                        )
            
            # Automatic checkpointing (Generation 2 feature)
            await self._handle_automatic_checkpointing()
            
            return result
            
        except Exception as e:
            logger.error(f"Training step {self.step} failed: {e}")
            if self.resilience_manager and FailureType:
                recovery_plan = await self.resilience_manager.handle_training_failure(
                    FailureType.TRAINING, e, {'step': self.step, 'epoch': self.epoch}
                )
                
                # If recovery suggests aborting, raise the exception
                if recovery_plan.recovery_mode.value == 'abort':
                    raise
                
                # Otherwise, try to continue with resilience recovery
                logger.info(f"Attempting recovery from training failure")
            
            raise
    
    async def _execute_training_step_with_resilience(self, batch: Any, **kwargs) -> Any:
        """Execute training step with resilience management."""
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                return await self._execute_training_step(batch, **kwargs)
            
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    raise
                
                # Exponential backoff
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Training step failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                await asyncio.sleep(delay)
    
    async def _handle_automatic_checkpointing(self) -> None:
        """Handle automatic checkpointing based on configuration."""
        if not self.resilience_manager:
            return
        
        checkpoint_interval = getattr(self.config, 'checkpoint_interval', 100)
        
        # Checkpoint at regular intervals
        if self.step % checkpoint_interval == 0:
            await self._create_resilient_checkpoint()
    
    async def _create_resilient_checkpoint(self) -> Optional[str]:
        """Create checkpoint using resilience manager."""
        if not self.resilience_manager:
            return None
        
        try:
            # Get model and optimizer states
            model_state = None
            optimizer_state = None
            
            if hasattr(self.model, 'state_dict'):
                model_state = self.model.state_dict()
            elif self.model:
                model_state = self.model
            
            if hasattr(self.optimizer, 'state_dict'):
                optimizer_state = self.optimizer.state_dict()
            elif self.optimizer:
                optimizer_state = self.optimizer
            
            # Create checkpoint
            checkpoint_id = await self.resilience_manager.create_checkpoint(
                step=self.step,
                epoch=self.epoch,
                model_state=model_state,
                optimizer_state=optimizer_state,
                metrics=self.get_carbon_metrics()
            )
            
            logger.info(f"Created resilient checkpoint {checkpoint_id} at step {self.step}")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to create resilient checkpoint: {e}")
            if FailureType:
                await self.resilience_manager.handle_training_failure(
                    FailureType.STORAGE, e, {'step': self.step}
                )
            return None
    
    async def _on_training_failure(
        self, 
        failure_type,
        error: Exception,
        context: Dict[str, Any]
    ) -> None:
        """Handle training failure callback from resilience manager."""
        logger.warning(f"Training failure detected: {failure_type.value if hasattr(failure_type, 'value') else failure_type}")
        
        # Could implement additional failure handling logic here
        # Such as notifying external monitoring systems, adjusting training parameters, etc.
    
    async def _wait_for_resume(self) -> None:
        """Wait while training is paused."""
        while self._should_pause and not self._force_stop:
            await asyncio.sleep(1)
    
    async def _execute_training_step(self, batch: Any, **kwargs) -> Any:
        """Execute the actual training step with intelligent adaptation.
        
        This method now includes adaptive batch sizing and dynamic optimization
        based on current carbon conditions.
        """
        # Get current carbon intensity for adaptive training
        current_intensity = None
        if self.monitor:
            current_intensity = await self.monitor.get_current_intensity(self.region)
        
        # Adaptive training based on carbon intensity (Generation 1)
        adapted_kwargs = kwargs.copy()
        if current_intensity and hasattr(self, 'adaptive_training'):
            adapted_kwargs = self._adapt_training_params(
                current_intensity.carbon_intensity, 
                kwargs
            )
        
        # Apply Generation 3 performance optimizations
        if self.performance_optimizer:
            # Adapt performance profile based on current conditions
            resource_pressure = 0.5  # Could be calculated from actual metrics
            carbon_value = current_intensity.carbon_intensity if current_intensity else 100.0
            self.performance_optimizer.adapt_profile(carbon_value, resource_pressure)
            
            # Optimize training parameters
            adapted_kwargs = self.performance_optimizer.optimize_training_params(adapted_kwargs)
        
        # Execute training with different approaches based on model type
        if hasattr(self.model, 'train_step'):
            return await self._async_safe_call(
                self.model.train_step, batch, **adapted_kwargs
            )
        elif hasattr(self.model, 'forward'):
            # PyTorch-style training
            if self.optimizer:
                self.optimizer.zero_grad()
                output = self.model.forward(batch)
                if hasattr(output, 'backward'):
                    output.backward()
                self.optimizer.step()
                return output
            else:
                return self.model.forward(batch)
        elif hasattr(self.model, '__call__'):
            return await self._async_safe_call(self.model, batch, **adapted_kwargs)
        else:
            raise NotImplementedError(
                "No training step implementation found. "
                "Use framework-specific integrations or override _execute_training_step."
            )
    
    async def _async_safe_call(self, func, *args, **kwargs):
        """Safely call potentially async functions."""
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in training step: {e}")
            raise
    
    def _adapt_training_params(self, carbon_intensity: float, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt training parameters based on carbon intensity."""
        adapted = kwargs.copy()
        
        # Reduce computational intensity during high carbon periods
        if carbon_intensity > self.config.pause_threshold * 0.8:
            # High carbon: reduce batch size, lower precision
            if 'batch_size' in adapted:
                adapted['batch_size'] = max(1, adapted['batch_size'] // 2)
            adapted['low_carbon_mode'] = True
        elif carbon_intensity < self.config.resume_threshold * 1.2:
            # Low carbon: can increase computational intensity
            if 'batch_size' in adapted:
                adapted['batch_size'] = min(
                    adapted.get('max_batch_size', adapted['batch_size'] * 2),
                    adapted['batch_size'] * 2
                )
            adapted['low_carbon_mode'] = False
        
        return adapted
    
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
        """Get comprehensive carbon and training metrics with predictions.
        
        Returns:
            Dictionary with current metrics and intelligent predictions
        """
        runtime = datetime.now() - self.metrics.start_time
        
        metrics = {
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
            'current_intensity': None,
            'efficiency_score': self._calculate_efficiency_score(),
            'carbon_trend': 'stable',
            'next_optimal_window': None,
            'estimated_completion_time': None,
            'carbon_budget_remaining': None,
            # Generation 2 metrics
            'resilience_active': self.resilience_manager is not None,
            'checkpoints_created': len(self.resilience_manager.checkpoints) if self.resilience_manager else 0,
            # Generation 3 metrics
            'performance_optimization_active': self.performance_optimizer is not None,
            'performance_stats': self.performance_optimizer.get_performance_stats() if self.performance_optimizer else None
        }
        
        # Add real-time current intensity if monitor available
        if self.monitor:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Can't await in sync method, use last known value
                    metrics['current_intensity'] = getattr(self, '_last_intensity', None)
                else:
                    current = loop.run_until_complete(
                        self.monitor.get_current_intensity(self.region)
                    )
                    if current:
                        metrics['current_intensity'] = current.carbon_intensity
                        self._last_intensity = current.carbon_intensity
            except Exception as e:
                logger.debug(f"Could not get current intensity: {e}")
        
        return metrics
    
    def _calculate_efficiency_score(self) -> float:
        """Calculate training efficiency score (0-100)."""
        if self.step == 0:
            return 0.0
        
        # Base efficiency on carbon vs time trade-off
        runtime_hours = (datetime.now() - self.metrics.start_time).total_seconds() / 3600
        pause_ratio = self.metrics.paused_duration.total_seconds() / 3600 / max(runtime_hours, 0.01)
        
        # Higher score for lower carbon with reasonable pause ratio
        carbon_efficiency = max(0, 100 - (self.metrics.avg_carbon_intensity / 10))
        time_efficiency = max(0, 100 - (pause_ratio * 200))  # Penalize excessive pausing
        
        return (carbon_efficiency * 0.7 + time_efficiency * 0.3)
    
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
    
    async def intelligent_training_plan(
        self,
        total_steps: int,
        estimated_duration_hours: float
    ) -> Dict[str, Any]:
        """Create intelligent training plan based on carbon forecasts.
        
        Args:
            total_steps: Expected total training steps
            estimated_duration_hours: Estimated training duration
            
        Returns:
            Comprehensive training plan with carbon optimization
        """
        if not self.monitor:
            return {"error": "Monitor not initialized"}
        
        # Get carbon forecast for planning
        forecast = await self.monitor.get_forecast(
            self.region, 
            hours=max(48, int(estimated_duration_hours * 1.5))
        )
        
        # Find optimal training windows
        optimal_windows = []
        remaining_hours = estimated_duration_hours
        current_time = datetime.now()
        
        while remaining_hours > 0 and len(optimal_windows) < 10:  # Max 10 windows
            window = await self.find_optimal_training_window(
                duration_hours=min(8, remaining_hours),  # Max 8-hour chunks
                start_time=current_time
            )
            
            if window:
                optimal_windows.append({
                    'start': window.start_time.isoformat(),
                    'end': window.end_time.isoformat(),
                    'duration_hours': (window.end_time - window.start_time).total_seconds() / 3600,
                    'avg_carbon_intensity': window.avg_carbon_intensity,
                    'expected_carbon_kg': window.total_expected_carbon_kg
                })
                remaining_hours -= (window.end_time - window.start_time).total_seconds() / 3600
                current_time = window.end_time + timedelta(hours=1)  # Buffer between windows
            else:
                break
        
        # Calculate carbon savings vs baseline
        baseline_carbon = estimated_duration_hours * 0.4 * 500  # Assume 400W at 500g/kWh baseline
        optimized_carbon = sum(w['expected_carbon_kg'] for w in optimal_windows)
        
        return {
            'total_estimated_steps': total_steps,
            'estimated_duration_hours': estimated_duration_hours,
            'optimal_windows': optimal_windows,
            'total_windows': len(optimal_windows),
            'baseline_carbon_kg': baseline_carbon,
            'optimized_carbon_kg': optimized_carbon,
            'carbon_savings_kg': baseline_carbon - optimized_carbon,
            'carbon_reduction_percent': ((baseline_carbon - optimized_carbon) / baseline_carbon * 100) if baseline_carbon > 0 else 0,
            'recommended_start_time': optimal_windows[0]['start'] if optimal_windows else None,
            'completion_time': optimal_windows[-1]['end'] if optimal_windows else None
        }
    
    async def adaptive_batch_scheduler(self, total_batches: int) -> List[Dict[str, Any]]:
        """Create adaptive batch schedule based on carbon forecasts.
        
        Args:
            total_batches: Total number of batches to process
            
        Returns:
            List of batch execution plans with carbon optimization
        """
        if not self.monitor:
            return []
        
        # Get hourly carbon forecast
        forecast = await self.monitor.get_forecast(self.region, hours=72)
        if not forecast or not forecast.data_points:
            return []
        
        # Create adaptive batch schedule
        batch_schedule = []
        batches_per_hour = max(1, total_batches // 24)  # Distribute over 24 hours
        
        for i, intensity_point in enumerate(forecast.data_points[:24]):  # Next 24 hours
            hour_start = intensity_point.timestamp
            hour_end = hour_start + timedelta(hours=1)
            
            # Adapt batch count based on carbon intensity
            if intensity_point.carbon_intensity < self.config.resume_threshold:
                # Low carbon: process more batches
                hour_batches = min(batches_per_hour * 2, total_batches)
            elif intensity_point.carbon_intensity > self.config.pause_threshold:
                # High carbon: process fewer batches or skip
                hour_batches = max(0, batches_per_hour // 3)
            else:
                # Normal carbon: standard batch count
                hour_batches = batches_per_hour
            
            if hour_batches > 0:
                batch_schedule.append({
                    'hour': i,
                    'start_time': hour_start.isoformat(),
                    'end_time': hour_end.isoformat(),
                    'batch_count': hour_batches,
                    'carbon_intensity': intensity_point.carbon_intensity,
                    'priority': 'high' if intensity_point.carbon_intensity < 50 else 'normal',
                    'estimated_carbon_kg': hour_batches * 0.001 * intensity_point.carbon_intensity
                })
                
                total_batches -= hour_batches
                if total_batches <= 0:
                    break
        
        return batch_schedule
    
    @asynccontextmanager
    async def training_session(self):
        """Enhanced context manager for carbon-aware training session."""
        await self.start_training()
        try:
            yield self
        finally:
            await self.stop_training()