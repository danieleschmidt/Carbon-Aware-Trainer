"""Comprehensive metrics collection and tracking."""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import threading
from collections import defaultdict, deque

from ..core.types import TrainingMetrics, CarbonIntensity
from ..core.exceptions import MetricsError
from ..core.power import PowerReading


logger = logging.getLogger(__name__)


@dataclass
class TrainingStep:
    """Individual training step metrics."""
    step: int
    timestamp: datetime
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    power_watts: Optional[float] = None
    carbon_intensity: Optional[float] = None
    carbon_emissions_g: Optional[float] = None
    duration_ms: Optional[float] = None
    gpu_utilization: Optional[float] = None
    memory_usage_mb: Optional[int] = None


@dataclass
class EpochSummary:
    """Epoch-level summary metrics."""
    epoch: int
    start_time: datetime
    end_time: datetime
    steps: int
    avg_loss: float
    avg_accuracy: Optional[float] = None
    total_energy_kwh: float = 0.0
    total_carbon_kg: float = 0.0
    avg_carbon_intensity: float = 0.0
    paused_duration_seconds: float = 0.0
    avg_power_watts: Optional[float] = None


@dataclass 
class SessionSummary:
    """Complete training session summary."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_steps: int = 0
    total_epochs: int = 0
    total_energy_kwh: float = 0.0
    total_carbon_kg: float = 0.0
    avg_carbon_intensity: float = 0.0
    peak_carbon_intensity: float = 0.0
    min_carbon_intensity: float = float('inf')
    total_paused_duration: timedelta = timedelta(0)
    carbon_savings_kg: float = 0.0
    final_loss: Optional[float] = None
    final_accuracy: Optional[float] = None
    regions_used: List[str] = None
    migrations: int = 0


class MetricsCollector:
    """Comprehensive metrics collection and management."""
    
    def __init__(
        self,
        session_id: str,
        export_path: Optional[Path] = None,
        real_time_export: bool = False,
        max_memory_steps: int = 10000
    ):
        """Initialize metrics collector.
        
        Args:
            session_id: Unique session identifier
            export_path: Path to export metrics
            real_time_export: Whether to export metrics in real-time
            max_memory_steps: Maximum steps to keep in memory
        """
        self.session_id = session_id
        self.export_path = export_path
        self.real_time_export = real_time_export
        self.max_memory_steps = max_memory_steps
        
        # Metrics storage       
        self.training_steps: deque = deque(maxlen=max_memory_steps)
        self.epoch_summaries: List[EpochSummary] = []
        self.session_summary = SessionSummary(
            session_id=session_id,
            start_time=datetime.now(),
            regions_used=[]
        )
        
        # Real-time tracking
        self._current_epoch = 0
        self._current_step = 0
        self._epoch_start_time: Optional[datetime] = None
        self._epoch_steps: List[TrainingStep] = []
        self._session_start_time = datetime.now()
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Performance tracking
        self._step_times: deque = deque(maxlen=100)  # Last 100 step times
        self._power_readings: deque = deque(maxlen=1000)  # Last 1000 power readings
        
        # Carbon tracking
        self._carbon_history: List[tuple] = []  # (timestamp, intensity, region)
        self._baseline_emissions: Optional[float] = None
    
    def start_epoch(self, epoch: int) -> None:
        """Start tracking a new epoch.
        
        Args:
            epoch: Epoch number
        """
        with self._lock:
            self._current_epoch = epoch
            self._epoch_start_time = datetime.now()
            self._epoch_steps = []
            
            logger.debug(f"Started tracking epoch {epoch}")
    
    def log_training_step(
        self,
        step: int,
        loss: Optional[float] = None,
        accuracy: Optional[float] = None,
        learning_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
        power_watts: Optional[float] = None,
        carbon_intensity: Optional[float] = None,
        duration_ms: Optional[float] = None,
        gpu_utilization: Optional[float] = None,
        memory_usage_mb: Optional[int] = None,
        **kwargs
    ) -> None:
        """Log metrics for a training step.
        
        Args:
            step: Step number
            loss: Training loss
            accuracy: Training accuracy
            learning_rate: Current learning rate
            batch_size: Batch size used
            power_watts: Power consumption
            carbon_intensity: Grid carbon intensity
            duration_ms: Step duration in milliseconds
            gpu_utilization: GPU utilization percentage
            memory_usage_mb: Memory usage in MB
            **kwargs: Additional metrics
        """
        timestamp = datetime.now()
        
        # Calculate carbon emissions if both power and intensity available
        carbon_emissions_g = None
        if power_watts and carbon_intensity:
            # Convert power to energy (very rough approximation)
            energy_kwh = (power_watts * (duration_ms or 1000) / 1000) / 3600000
            carbon_emissions_g = energy_kwh * carbon_intensity
        
        step_metrics = TrainingStep(
            step=step,
            timestamp=timestamp,
            loss=loss,
            accuracy=accuracy,
            learning_rate=learning_rate,
            batch_size=batch_size,
            power_watts=power_watts,
            carbon_intensity=carbon_intensity,
            carbon_emissions_g=carbon_emissions_g,
            duration_ms=duration_ms,
            gpu_utilization=gpu_utilization,
            memory_usage_mb=memory_usage_mb
        )
        
        # Add any additional metrics
        for key, value in kwargs.items():
            if hasattr(step_metrics, key):
                setattr(step_metrics, key, value)
        
        with self._lock:
            self.training_steps.append(step_metrics)
            self._epoch_steps.append(step_metrics)
            self._current_step = step
            
            # Track step timing
            if duration_ms:
                self._step_times.append(duration_ms)
            
            # Update session summary
            self.session_summary.total_steps = step
            if carbon_emissions_g:
                self.session_summary.total_carbon_kg += carbon_emissions_g / 1000
            
            if carbon_intensity:
                # Update carbon intensity stats
                if carbon_intensity > self.session_summary.peak_carbon_intensity:
                    self.session_summary.peak_carbon_intensity = carbon_intensity
                if carbon_intensity < self.session_summary.min_carbon_intensity:
                    self.session_summary.min_carbon_intensity = carbon_intensity
                
                # Update running average
                self.session_summary.avg_carbon_intensity = (
                    (self.session_summary.avg_carbon_intensity * (step - 1) + carbon_intensity) / step
                    if step > 0 else carbon_intensity
                )
        
        # Real-time export if enabled
        if self.real_time_export and self.export_path:
            self._export_step_metrics(step_metrics)
        
        logger.debug(f"Logged training step {step}")
    
    def end_epoch(self, epoch: int) -> EpochSummary:
        """Finish tracking an epoch and generate summary.
        
        Args:
            epoch: Epoch number
            
        Returns:
            Epoch summary metrics
        """
        with self._lock:
            if not self._epoch_start_time or not self._epoch_steps:
                raise MetricsError(f"No data available for epoch {epoch}")
            
            end_time = datetime.now()
            
            # Calculate epoch statistics
            losses = [s.loss for s in self._epoch_steps if s.loss is not None]
            accuracies = [s.accuracy for s in self._epoch_steps if s.accuracy is not None]
            carbon_emissions = [s.carbon_emissions_g for s in self._epoch_steps if s.carbon_emissions_g is not None]
            carbon_intensities = [s.carbon_intensity for s in self._epoch_steps if s.carbon_intensity is not None]
            power_readings = [s.power_watts for s in self._epoch_steps if s.power_watts is not None]
            
            epoch_summary = EpochSummary(
                epoch=epoch,
                start_time=self._epoch_start_time,
                end_time=end_time,
                steps=len(self._epoch_steps),
                avg_loss=sum(losses) / len(losses) if losses else 0.0,
                avg_accuracy=sum(accuracies) / len(accuracies) if accuracies else None,
                total_energy_kwh=0.0,  # TODO: Calculate from power readings
                total_carbon_kg=sum(carbon_emissions) / 1000 if carbon_emissions else 0.0,
                avg_carbon_intensity=sum(carbon_intensities) / len(carbon_intensities) if carbon_intensities else 0.0,
                avg_power_watts=sum(power_readings) / len(power_readings) if power_readings else None
            )
            
            self.epoch_summaries.append(epoch_summary)
            self.session_summary.total_epochs = epoch + 1
            
            # Update session summary
            if losses:
                self.session_summary.final_loss = losses[-1]
            if accuracies:
                self.session_summary.final_accuracy = accuracies[-1]
            
            logger.info(f"Completed epoch {epoch}: avg_loss={epoch_summary.avg_loss:.4f}")
            
            return epoch_summary
    
    def record_pause(self, duration: timedelta, reason: str = "") -> None:
        """Record training pause.
        
        Args:
            duration: Pause duration
            reason: Reason for pause
        """
        with self._lock:
            self.session_summary.total_paused_duration += duration
            
            logger.info(f"Recorded pause: {duration} ({reason})")
    
    def record_migration(self, from_region: str, to_region: str) -> None:
        """Record cross-region migration.
        
        Args:
            from_region: Source region
            to_region: Destination region
        """
        with self._lock:
            self.session_summary.migrations += 1
            
            if to_region not in self.session_summary.regions_used:
                self.session_summary.regions_used.append(to_region)
            
            logger.info(f"Recorded migration: {from_region} -> {to_region}")
    
    def record_carbon_intensity(self, intensity: CarbonIntensity) -> None:
        """Record carbon intensity measurement.
        
        Args:
            intensity: Carbon intensity reading
        """
        with self._lock:
            self._carbon_history.append((
                intensity.timestamp,
                intensity.carbon_intensity,
                intensity.region
            ))
            
            # Keep only recent history (last 24 hours)
            cutoff = datetime.now() - timedelta(hours=24)
            self._carbon_history = [
                h for h in self._carbon_history 
                if h[0] > cutoff
            ]
    
    def record_power_reading(self, reading: PowerReading) -> None:
        """Record power consumption reading.
        
        Args:
            reading: Power reading
        """
        with self._lock:
            self._power_readings.append(reading)
    
    def calculate_carbon_savings(self, baseline_emissions_kg: float) -> float:
        """Calculate carbon savings compared to baseline.
        
        Args:
            baseline_emissions_kg: Baseline emissions without carbon awareness
            
        Returns:
            Carbon savings in kg CO2
        """
        with self._lock:
            actual_emissions = self.session_summary.total_carbon_kg
            savings = max(0, baseline_emissions_kg - actual_emissions)
            
            self.session_summary.carbon_savings_kg = savings
            self._baseline_emissions = baseline_emissions_kg
            
            return savings
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        with self._lock:
            if not self._step_times:
                return {"error": "No timing data available"}
            
            step_times = list(self._step_times)
            
            return {
                "avg_step_time_ms": sum(step_times) / len(step_times),
                "min_step_time_ms": min(step_times),
                "max_step_time_ms": max(step_times),
                "total_steps": len(self.training_steps),
                "steps_per_second": 1000 / (sum(step_times) / len(step_times)) if step_times else 0
            }
    
    def get_carbon_stats(self) -> Dict[str, Any]:
        """Get carbon emission statistics.
        
        Returns:
            Dictionary with carbon metrics
        """
        with self._lock:
            return {
                "total_carbon_kg": self.session_summary.total_carbon_kg,
                "avg_carbon_intensity": self.session_summary.avg_carbon_intensity,
                "peak_carbon_intensity": self.session_summary.peak_carbon_intensity,
                "min_carbon_intensity": (
                    self.session_summary.min_carbon_intensity 
                    if self.session_summary.min_carbon_intensity != float('inf') 
                    else None
                ),
                "carbon_savings_kg": self.session_summary.carbon_savings_kg,
                "baseline_emissions_kg": self._baseline_emissions,
                "total_paused_hours": self.session_summary.total_paused_duration.total_seconds() / 3600,
                "regions_used": self.session_summary.regions_used,
                "migrations": self.session_summary.migrations
            }
    
    def get_session_summary(self) -> SessionSummary:
        """Get complete session summary.
        
        Returns:
            Session summary with all metrics
        """
        with self._lock:
            # Update end time if session is complete
            if self.session_summary.end_time is None:
                self.session_summary.end_time = datetime.now()
            
            return self.session_summary
    
    def export_metrics(self, path: Optional[Path] = None) -> bool:
        """Export all metrics to file.
        
        Args:
            path: Export path (uses default if None)
            
        Returns:
            True if export successful
        """
        export_path = path or self.export_path
        if not export_path:
            logger.warning("No export path specified")
            return False
        
        try:
            export_data = {
                "session_summary": asdict(self.get_session_summary()),
                "epoch_summaries": [asdict(e) for e in self.epoch_summaries],
                "training_steps": [asdict(s) for s in list(self.training_steps)],
                "carbon_history": [
                    {"timestamp": t.isoformat(), "intensity": i, "region": r}
                    for t, i, r in self._carbon_history
                ],
                "export_timestamp": datetime.now().isoformat()
            }
            
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported metrics to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return False
    
    def _export_step_metrics(self, step: TrainingStep) -> None:
        """Export individual step metrics in real-time.
        
        Args:
            step: Training step to export
        """
        if not self.export_path:
            return
        
        try:
            step_file = self.export_path.parent / f"{self.session_id}_steps.jsonl"
            
            with open(step_file, 'a') as f:
                json.dump(asdict(step), f, default=str)
                f.write('\n')
                
        except Exception as e:
            logger.debug(f"Failed to export step metrics: {e}")
    
    def get_recent_steps(self, count: int = 100) -> List[TrainingStep]:
        """Get recent training steps.
        
        Args:
            count: Number of recent steps to return
            
        Returns:
            List of recent training steps
        """
        with self._lock:
            return list(self.training_steps)[-count:]
    
    def get_carbon_trend(self, hours: int = 1) -> List[tuple]:
        """Get carbon intensity trend.
        
        Args:
            hours: Hours of history to return
            
        Returns:
            List of (timestamp, intensity, region) tuples
        """
        with self._lock:
            cutoff = datetime.now() - timedelta(hours=hours)
            return [
                h for h in self._carbon_history 
                if h[0] > cutoff
            ]