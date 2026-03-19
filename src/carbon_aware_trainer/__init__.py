"""Carbon-aware ML training scheduler.

Shifts training jobs to low-carbon time windows by tracking grid carbon intensity
and finding optimal scheduling opportunities.

Quick start:
    from carbon_aware_trainer import CarbonIntensityProvider, TrainingScheduler, CarbonAwareCallback

    provider = CarbonIntensityProvider(region="US-CA")
    scheduler = TrainingScheduler(provider)
    window = scheduler.find_optimal_window(job_hours=6.0, lookahead_hours=48)
    print(f"Start at {window.start_time} — saves {window.carbon_saved_kg:.2f} kg CO2")
"""

from .provider import CarbonIntensityProvider, CarbonReading
from .scheduler import TrainingScheduler, TrainingJob, ScheduledWindow
from .callback import CarbonAwareCallback
from .energy import EnergyEstimator

__all__ = [
    "CarbonIntensityProvider",
    "CarbonReading",
    "TrainingScheduler",
    "TrainingJob",
    "ScheduledWindow",
    "CarbonAwareCallback",
    "EnergyEstimator",
]

__version__ = "2.0.0"
