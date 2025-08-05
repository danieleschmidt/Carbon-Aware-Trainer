"""Simple threshold-based carbon-aware scheduling strategy."""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging

from ..core.types import TrainingState, CarbonIntensity, OptimalWindow
from ..core.monitor import CarbonMonitor


logger = logging.getLogger(__name__)


class ThresholdScheduler:
    """Simple threshold-based scheduler for carbon-aware training."""
    
    def __init__(
        self, 
        carbon_threshold: float = 100.0,
        pause_threshold: Optional[float] = None,
        resume_threshold: Optional[float] = None,
        max_pause_duration: timedelta = timedelta(hours=6)
    ):
        """Initialize threshold scheduler.
        
        Args:
            carbon_threshold: Base carbon intensity threshold (gCO2/kWh)
            pause_threshold: Threshold to pause training (defaults to 1.5x base)
            resume_threshold: Threshold to resume training (defaults to 0.8x base)
            max_pause_duration: Maximum pause duration before forcing resume
        """
        self.carbon_threshold = carbon_threshold
        self.pause_threshold = pause_threshold or (carbon_threshold * 1.5)
        self.resume_threshold = resume_threshold or (carbon_threshold * 0.8)
        self.max_pause_duration = max_pause_duration
        
        # State tracking
        self._last_decision_time = datetime.now()
        self._pause_start_time: Optional[datetime] = None
        self._decision_history: List[Dict[str, Any]] = []
    
    def should_pause_training(self, intensity: CarbonIntensity) -> bool:
        """Determine if training should be paused.
        
        Args:
            intensity: Current carbon intensity
            
        Returns:
            True if training should be paused
        """
        carbon_value = intensity.carbon_intensity
        should_pause = carbon_value > self.pause_threshold
        
        # Log decision
        decision = {
            'timestamp': datetime.now(),
            'carbon_intensity': carbon_value,
            'threshold': self.pause_threshold,
            'decision': 'pause' if should_pause else 'continue',
            'region': intensity.region
        }
        self._decision_history.append(decision)
        
        # Keep only recent decisions
        if len(self._decision_history) > 100:
            self._decision_history = self._decision_history[-100:]
        
        if should_pause:
            logger.info(
                f"Recommending pause: carbon intensity {carbon_value:.1f} "
                f"> threshold {self.pause_threshold:.1f} gCO2/kWh"
            )
        
        return should_pause
    
    def should_resume_training(self, intensity: CarbonIntensity) -> bool:
        """Determine if training should be resumed.
        
        Args:
            intensity: Current carbon intensity
            
        Returns:
            True if training should be resumed
        """
        carbon_value = intensity.carbon_intensity
        
        # Check carbon intensity threshold
        below_threshold = carbon_value < self.resume_threshold
        
        # Check maximum pause duration
        force_resume = False
        if self._pause_start_time:
            pause_duration = datetime.now() - self._pause_start_time
            force_resume = pause_duration > self.max_pause_duration
        
        should_resume = below_threshold or force_resume
        
        # Log decision
        decision = {
            'timestamp': datetime.now(),
            'carbon_intensity': carbon_value,
            'threshold': self.resume_threshold,
            'decision': 'resume' if should_resume else 'wait',
            'region': intensity.region,
            'force_resume': force_resume
        }
        self._decision_history.append(decision)
        
        if should_resume:
            reason = "max pause duration exceeded" if force_resume else f"carbon intensity {carbon_value:.1f} < threshold {self.resume_threshold:.1f}"
            logger.info(f"Recommending resume: {reason}")
        
        return should_resume
    
    def on_training_paused(self) -> None:
        """Called when training is paused."""
        self._pause_start_time = datetime.now()
        logger.debug("Training paused - starting pause timer")
    
    def on_training_resumed(self) -> None:
        """Called when training is resumed."""
        if self._pause_start_time:
            pause_duration = datetime.now() - self._pause_start_time
            logger.info(f"Training resumed after {pause_duration}")
        
        self._pause_start_time = None
    
    async def find_next_training_window(
        self,
        monitor: CarbonMonitor,
        region: str,
        duration_hours: int = 8,
        max_wait_hours: int = 48
    ) -> Optional[OptimalWindow]:
        """Find the next suitable training window.
        
        Args:
            monitor: Carbon monitor for forecast data
            region: Training region
            duration_hours: Required training duration
            max_wait_hours: Maximum time to look ahead
            
        Returns:
            Next optimal training window or None
        """
        forecast = await monitor.get_forecast(region, hours=max_wait_hours)
        
        if not forecast or not forecast.data_points:
            logger.warning(f"No forecast available for {region}")
            return None
        
        # Find first contiguous window where carbon stays below threshold
        data_points = sorted(forecast.data_points, key=lambda x: x.timestamp)
        
        for i in range(len(data_points) - duration_hours + 1):
            window_points = data_points[i:i + duration_hours]
            
            # Check if all points in window are below threshold
            max_intensity = max(p.carbon_intensity for p in window_points)
            avg_intensity = sum(p.carbon_intensity for p in window_points) / len(window_points)
            
            if max_intensity <= self.carbon_threshold:
                # Found suitable window
                renewable_pct = sum(
                    p.renewable_percentage or 0 
                    for p in window_points
                ) / len(window_points)
                
                return OptimalWindow(
                    start_time=window_points[0].timestamp,
                    end_time=window_points[-1].timestamp,
                    avg_carbon_intensity=avg_intensity,
                    total_expected_carbon_kg=0.0,  # To be calculated with actual consumption
                    confidence_score=0.8,  # Simple threshold has moderate confidence
                    renewable_percentage=renewable_pct,
                    region=region
                )
        
        logger.warning(
            f"No suitable {duration_hours}h training window found "
            f"within {max_wait_hours}h for threshold {self.carbon_threshold} gCO2/kWh"
        )
        return None
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get statistics about scheduling decisions.
        
        Returns:
            Dictionary with strategy performance stats
        """
        if not self._decision_history:
            return {"decisions": 0}
        
        recent_decisions = [
            d for d in self._decision_history
            if (datetime.now() - d['timestamp']).total_seconds() < 86400  # Last 24h
        ]
        
        pause_decisions = len([d for d in recent_decisions if d['decision'] == 'pause'])
        resume_decisions = len([d for d in recent_decisions if d['decision'] == 'resume'])
        
        avg_carbon_when_paused = 0
        avg_carbon_when_resumed = 0
        
        pause_intensities = [d['carbon_intensity'] for d in recent_decisions if d['decision'] == 'pause']
        resume_intensities = [d['carbon_intensity'] for d in recent_decisions if d['decision'] == 'resume']
        
        if pause_intensities:
            avg_carbon_when_paused = sum(pause_intensities) / len(pause_intensities)
        if resume_intensities:
            avg_carbon_when_resumed = sum(resume_intensities) / len(resume_intensities)
        
        return {
            "strategy": "threshold",
            "carbon_threshold": self.carbon_threshold,
            "pause_threshold": self.pause_threshold,
            "resume_threshold": self.resume_threshold,
            "decisions_24h": len(recent_decisions),
            "pause_decisions_24h": pause_decisions,
            "resume_decisions_24h": resume_decisions,
            "avg_carbon_when_paused": avg_carbon_when_paused,
            "avg_carbon_when_resumed": avg_carbon_when_resumed,
            "current_pause_duration": (
                (datetime.now() - self._pause_start_time).total_seconds() / 3600
                if self._pause_start_time else 0
            )
        }