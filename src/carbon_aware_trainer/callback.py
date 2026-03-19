"""CarbonAwareCallback: pause/resume training based on real-time carbon intensity.

Drop-in callback for PyTorch training loops. Checks carbon intensity at the
start of each epoch (or every N steps) and either:
  - Continues normally if intensity is below threshold.
  - Pauses (sleeps) until intensity drops back to the resume threshold.
  - Logs carbon consumed so far.

Usage with a plain PyTorch loop::

    from carbon_aware_trainer import CarbonAwareCallback, CarbonIntensityProvider

    provider = CarbonIntensityProvider(region="US-CA")
    cb = CarbonAwareCallback(provider, threshold_gco2_kwh=250, gpu_tdp_watts=300)

    for epoch in range(num_epochs):
        cb.on_epoch_start(epoch)
        for batch in dataloader:
            loss = train_step(batch)
            cb.on_step_end()

    print(cb.summary())

Usage with PyTorch Lightning (via on_train_epoch_start)::

    import pytorch_lightning as pl

    class MyModule(pl.LightningModule, CarbonAwareCallback):
        def __init__(self):
            super().__init__()
            CarbonAwareCallback.__init__(self, provider)

        def on_train_epoch_start(self):
            self.on_epoch_start(self.current_epoch)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

from .provider import CarbonIntensityProvider, CarbonReading
from .energy import EnergyEstimator

logger = logging.getLogger(__name__)


class CarbonAwareCallback:
    """Pause/resume training based on live carbon intensity.

    Args:
        provider: CarbonIntensityProvider to poll for readings.
        threshold_gco2_kwh: Intensity above which training pauses.
        resume_threshold_gco2_kwh: Intensity below which paused training resumes.
            Defaults to 90% of `threshold_gco2_kwh` (hysteresis).
        gpu_tdp_watts: GPU power draw for carbon accounting.
        check_interval_seconds: How often to re-check intensity while paused.
        max_pause_seconds: Maximum pause duration before resuming anyway (safety
            valve). 0 = no limit.
        dry_run: Log decisions but never actually sleep (useful for testing).
    """

    def __init__(
        self,
        provider: CarbonIntensityProvider,
        threshold_gco2_kwh: float = 300.0,
        resume_threshold_gco2_kwh: Optional[float] = None,
        gpu_tdp_watts: float = 300.0,
        check_interval_seconds: int = 60,
        max_pause_seconds: int = 0,
        dry_run: bool = False,
    ) -> None:
        self.provider = provider
        self.threshold = threshold_gco2_kwh
        self.resume_threshold = resume_threshold_gco2_kwh or threshold_gco2_kwh * 0.9
        self.gpu_tdp_watts = gpu_tdp_watts
        self.check_interval = check_interval_seconds
        self.max_pause_seconds = max_pause_seconds
        self.dry_run = dry_run

        # Accounting
        self._training_start: Optional[datetime] = None
        self._active_seconds: float = 0.0
        self._paused_seconds: float = 0.0
        self._step_start: Optional[float] = None
        self._steps: int = 0
        self._pauses: int = 0
        self._readings: list[CarbonReading] = []

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_train_start(self) -> None:
        """Call once before the training loop begins."""
        self._training_start = datetime.utcnow()
        self._step_start = time.monotonic()
        reading = self.provider.current()
        self._readings.append(reading)
        logger.info(
            f"[CarbonAwareCallback] Training started. "
            f"Current intensity: {reading.intensity_gco2_kwh:.0f} gCO2/kWh "
            f"(threshold: {self.threshold:.0f})"
        )

    def on_epoch_start(self, epoch: int) -> None:
        """Call at the start of each epoch. Pauses if intensity is too high."""
        self._check_and_maybe_pause(context=f"epoch {epoch}")

    def on_step_end(self) -> None:
        """Call after each training step. Tracks active time."""
        self._steps += 1
        if self._step_start is not None:
            elapsed = time.monotonic() - self._step_start
            self._active_seconds += elapsed
            self._step_start = time.monotonic()

    def on_train_end(self) -> None:
        """Call when training finishes. Logs a final summary."""
        logger.info(self.summary())

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _check_and_maybe_pause(self, context: str = "") -> None:
        """Check current intensity and sleep until it drops if needed."""
        reading = self.provider.current()
        self._readings.append(reading)
        intensity = reading.intensity_gco2_kwh

        if intensity <= self.threshold:
            logger.debug(
                f"[CarbonAwareCallback] {context}: "
                f"{intensity:.0f} gCO2/kWh ≤ threshold {self.threshold:.0f} — continue"
            )
            return

        # Intensity too high — pause
        self._pauses += 1
        pause_start = time.monotonic()
        pause_wall = datetime.utcnow()

        logger.warning(
            f"[CarbonAwareCallback] {context}: "
            f"{intensity:.0f} gCO2/kWh > threshold {self.threshold:.0f} — "
            f"{'[DRY RUN] ' if self.dry_run else ''}pausing until ≤ {self.resume_threshold:.0f}"
        )

        if self.dry_run:
            return

        while True:
            time.sleep(self.check_interval)
            reading = self.provider.current()
            self._readings.append(reading)
            intensity = reading.intensity_gco2_kwh
            elapsed_pause = time.monotonic() - pause_start

            logger.info(
                f"[CarbonAwareCallback] Still paused. "
                f"Intensity: {intensity:.0f} gCO2/kWh "
                f"(resume at ≤ {self.resume_threshold:.0f}). "
                f"Paused {elapsed_pause/60:.1f}m so far."
            )

            if intensity <= self.resume_threshold:
                self._paused_seconds += elapsed_pause
                logger.info(
                    f"[CarbonAwareCallback] Resuming — intensity dropped to "
                    f"{intensity:.0f} gCO2/kWh after {elapsed_pause/60:.1f}m pause."
                )
                break

            if self.max_pause_seconds and elapsed_pause >= self.max_pause_seconds:
                self._paused_seconds += elapsed_pause
                logger.warning(
                    f"[CarbonAwareCallback] Max pause ({self.max_pause_seconds}s) reached — "
                    f"resuming despite high intensity ({intensity:.0f} gCO2/kWh)."
                )
                break

        # Reset step timer after pause
        self._step_start = time.monotonic()

    # ------------------------------------------------------------------
    # Accounting / reporting
    # ------------------------------------------------------------------

    @property
    def total_runtime_hours(self) -> float:
        if self._training_start is None:
            return 0.0
        return (datetime.utcnow() - self._training_start).total_seconds() / 3600

    @property
    def active_training_hours(self) -> float:
        return self._active_seconds / 3600

    @property
    def paused_hours(self) -> float:
        return self._paused_seconds / 3600

    @property
    def mean_intensity(self) -> float:
        if not self._readings:
            return 0.0
        return sum(r.intensity_gco2_kwh for r in self._readings) / len(self._readings)

    @property
    def carbon_emitted_kg(self) -> float:
        """Estimated carbon for active training time (not counting pauses)."""
        energy_kwh = EnergyEstimator.energy_kwh(self.gpu_tdp_watts, self.active_training_hours)
        return EnergyEstimator.carbon_kg(energy_kwh, self.mean_intensity)

    @property
    def carbon_avoided_kg(self) -> float:
        """Carbon avoided by pausing during high-intensity periods.

        Estimated as: what the paused hours would have emitted at the
        intensity that triggered each pause.
        """
        if not self._readings:
            return 0.0
        # Approximate: use readings above threshold as the "avoided" signal
        high_readings = [r for r in self._readings if r.intensity_gco2_kwh > self.threshold]
        if not high_readings:
            return 0.0
        avg_high = sum(r.intensity_gco2_kwh for r in high_readings) / len(high_readings)
        energy_kwh = EnergyEstimator.energy_kwh(self.gpu_tdp_watts, self.paused_hours)
        return EnergyEstimator.carbon_kg(energy_kwh, avg_high)

    def summary(self) -> str:
        return (
            f"[CarbonAwareCallback] Training complete.\n"
            f"  Steps: {self._steps} | Pauses: {self._pauses}\n"
            f"  Active: {self.active_training_hours:.2f}h | "
            f"Paused: {self.paused_hours:.2f}h\n"
            f"  Mean intensity: {self.mean_intensity:.0f} gCO2/kWh\n"
            f"  Estimated carbon emitted: {self.carbon_emitted_kg*1000:.1f}g CO2\n"
            f"  Estimated carbon avoided by pausing: {self.carbon_avoided_kg*1000:.1f}g CO2"
        )
