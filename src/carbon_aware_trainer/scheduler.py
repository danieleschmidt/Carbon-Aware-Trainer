"""TrainingScheduler: finds the optimal low-carbon window for a training job.

Algorithm:
  1. Fetch a carbon forecast for the lookahead window (default 48h).
  2. Slide a window of `job_hours` duration across the forecast.
  3. For each candidate start time, compute the mean carbon intensity over
     the job window.
  4. Return the window with the minimum mean intensity alongside a naive
     (start-now) baseline, so the caller can show carbon savings.

Carbon footprint calculation:
  carbon_kg = energy_kwh × intensity_gco2_kwh / 1000
  energy_kwh = gpu_tdp_watts × job_hours / 1000
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

from .provider import CarbonIntensityProvider, CarbonReading
from .energy import EnergyEstimator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class TrainingJob:
    """Describes a training job for scheduling purposes.

    Args:
        job_hours: Estimated wall-clock training duration in hours.
        gpu_tdp_watts: GPU thermal design power in watts (default: 300W ≈ A100).
        name: Optional job name for display.
    """
    job_hours: float
    gpu_tdp_watts: float = 300.0
    name: str = "training-job"

    @property
    def energy_kwh(self) -> float:
        """Energy consumed at full TDP over the job duration."""
        return EnergyEstimator.energy_kwh(self.gpu_tdp_watts, self.job_hours)


@dataclass
class ScheduledWindow:
    """Result of a scheduling decision.

    Attributes:
        start_time: Recommended job start time (UTC).
        end_time: Expected job end time (UTC).
        mean_intensity_gco2_kwh: Mean carbon intensity during the window.
        carbon_kg: Estimated CO2 emissions for the job in this window.
        naive_start_time: "Start now" baseline start.
        naive_mean_intensity_gco2_kwh: Mean carbon intensity if started now.
        naive_carbon_kg: Estimated CO2 if started now.
        carbon_saved_kg: Absolute carbon savings vs naive.
        carbon_saved_pct: Percentage carbon savings vs naive.
        forecast_points: The carbon forecast readings used (one per hour).
    """
    start_time: datetime
    end_time: datetime
    mean_intensity_gco2_kwh: float
    carbon_kg: float

    naive_start_time: datetime
    naive_mean_intensity_gco2_kwh: float
    naive_carbon_kg: float

    carbon_saved_kg: float
    carbon_saved_pct: float

    forecast_points: List[CarbonReading]

    def summary(self) -> str:
        """Human-readable one-line summary."""
        delay_h = (self.start_time - self.naive_start_time).total_seconds() / 3600
        return (
            f"Optimal start: {self.start_time.strftime('%Y-%m-%d %H:%M UTC')}  "
            f"(+{delay_h:.1f}h delay)  |  "
            f"{self.mean_intensity_gco2_kwh:.0f} vs {self.naive_mean_intensity_gco2_kwh:.0f} gCO2/kWh  |  "
            f"saves {self.carbon_saved_kg*1000:.0f}g CO2 ({self.carbon_saved_pct:.1f}%)"
        )


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class TrainingScheduler:
    """Finds low-carbon training windows for ML jobs.

    Args:
        provider: CarbonIntensityProvider instance.
        lookahead_hours: How far ahead to search for an optimal window.
        resolution_minutes: Granularity of the forecast used for scheduling.
    """

    def __init__(
        self,
        provider: CarbonIntensityProvider,
        lookahead_hours: int = 48,
        resolution_minutes: int = 60,
    ) -> None:
        self.provider = provider
        self.lookahead_hours = lookahead_hours
        self.resolution_minutes = resolution_minutes

    def find_optimal_window(
        self,
        job: Optional[TrainingJob] = None,
        job_hours: Optional[float] = None,
        gpu_tdp_watts: float = 300.0,
    ) -> ScheduledWindow:
        """Find the optimal start window for a training job.

        You can pass either a `TrainingJob` object or raw `job_hours` / `gpu_tdp_watts`.

        Args:
            job: TrainingJob descriptor. If provided, `job_hours` and `gpu_tdp_watts`
                 are ignored.
            job_hours: Duration of the training job in hours.
            gpu_tdp_watts: GPU TDP in watts (used for energy estimation).

        Returns:
            ScheduledWindow with optimal start time and carbon savings estimate.
        """
        if job is None:
            if job_hours is None:
                raise ValueError("Either `job` or `job_hours` must be provided.")
            job = TrainingJob(job_hours=job_hours, gpu_tdp_watts=gpu_tdp_watts)

        # Fetch forecast
        forecast = self.provider.forecast(
            hours=self.lookahead_hours,
            resolution_minutes=self.resolution_minutes,
        )
        if not forecast:
            raise RuntimeError("No forecast data available from provider.")

        # Number of steps needed to cover the job duration
        steps_per_hour = 60 / self.resolution_minutes
        job_steps = max(1, round(job.job_hours * steps_per_hour))

        if job_steps > len(forecast):
            logger.warning(
                f"Job duration ({job.job_hours}h) exceeds forecast window "
                f"({self.lookahead_hours}h). Clamping search."
            )
            job_steps = len(forecast)

        # Sliding window: find the start index with minimum mean intensity
        best_idx = 0
        best_mean = float("inf")

        for i in range(len(forecast) - job_steps + 1):
            window = forecast[i : i + job_steps]
            mean_intensity = sum(r.intensity_gco2_kwh for r in window) / len(window)
            if mean_intensity < best_mean:
                best_mean = mean_intensity
                best_idx = i

        # Naive baseline: start now (index 0)
        naive_window = forecast[: job_steps]
        naive_mean = sum(r.intensity_gco2_kwh for r in naive_window) / len(naive_window)

        # Carbon calculations
        energy_kwh = job.energy_kwh
        optimal_carbon_kg = EnergyEstimator.carbon_kg(energy_kwh, best_mean)
        naive_carbon_kg = EnergyEstimator.carbon_kg(energy_kwh, naive_mean)
        saved_kg = naive_carbon_kg - optimal_carbon_kg
        saved_pct = (saved_kg / naive_carbon_kg * 100) if naive_carbon_kg > 0 else 0.0

        optimal_start = forecast[best_idx].timestamp
        optimal_end = optimal_start + timedelta(hours=job.job_hours)
        naive_start = forecast[0].timestamp

        return ScheduledWindow(
            start_time=optimal_start,
            end_time=optimal_end,
            mean_intensity_gco2_kwh=best_mean,
            carbon_kg=optimal_carbon_kg,
            naive_start_time=naive_start,
            naive_mean_intensity_gco2_kwh=naive_mean,
            naive_carbon_kg=naive_carbon_kg,
            carbon_saved_kg=saved_kg,
            carbon_saved_pct=saved_pct,
            forecast_points=forecast,
        )

    def hourly_carbon_profile(self, hours: int = 48) -> List[dict]:
        """Return a simple list of {time, intensity} dicts for the next N hours.

        Useful for displaying a carbon timeline.
        """
        forecast = self.provider.forecast(hours=hours, resolution_minutes=60)
        return [
            {
                "time": r.timestamp.strftime("%Y-%m-%d %H:%M UTC"),
                "intensity_gco2_kwh": round(r.intensity_gco2_kwh, 1),
            }
            for r in forecast
        ]
