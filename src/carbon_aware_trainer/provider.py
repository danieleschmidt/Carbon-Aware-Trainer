"""CarbonIntensityProvider: fetch or simulate grid carbon intensity (gCO2/kWh).

Two modes:
  - synthetic: 24-hour sinusoidal signal based on typical grid patterns — good
    for testing, demos, and offline use. No API key needed.
  - electricitymaps: ElectricityMaps API (requires ELECTRICITY_MAPS_API_KEY env
    var). Falls back to synthetic if the request fails.

Carbon intensity signal model (synthetic):
  Grid carbon intensity follows a rough 24h cycle in many regions:
    - Low at night (00:00-06:00): renewables dominate, low demand
    - Rising in morning (06:00-10:00): peak demand kicks in
    - High midday (10:00-16:00): maximum load, often gas/coal peakers
    - Solar dip (11:00-14:00 in sunny regions) reduces midday peak slightly
    - Evening peak (17:00-21:00): post-work demand spike
    - Dropping overnight: coal/gas units cycle off

  We model this as:
    base + peak_amp * sin(t) + noise
"""

from __future__ import annotations

import math
import os
import time
import random
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CarbonReading:
    """A single carbon intensity measurement."""
    timestamp: datetime
    intensity_gco2_kwh: float       # gCO2 per kWh
    region: str
    source: str = "synthetic"       # "synthetic" | "electricitymaps"
    is_forecast: bool = False

    def __repr__(self) -> str:
        label = "forecast" if self.is_forecast else "measured"
        return (
            f"CarbonReading({self.region} @ {self.timestamp.strftime('%H:%M')}: "
            f"{self.intensity_gco2_kwh:.0f} gCO2/kWh [{label}])"
        )


# ---------------------------------------------------------------------------
# Regional defaults for synthetic mode (annual average gCO2/kWh)
# Sources: ElectricityMaps / Ember data
# ---------------------------------------------------------------------------

REGION_BASELINES: Dict[str, float] = {
    "US-CA":  200,   # California — high solar, low gas
    "US-TX":  420,   # Texas — heavy gas, some wind
    "US-NY":  230,   # New York — nuclear + hydro heavy
    "US-MW":  580,   # US Midwest — coal belt
    "US-NE":  210,   # New England — nuclear + imports
    "EU-DE":  350,   # Germany — coal + wind mix
    "EU-FR":  55,    # France — nuclear dominant
    "EU-GB":  180,   # UK — offshore wind + gas
    "EU-PL":  750,   # Poland — coal heavy
    "AU-NSW": 700,   # New South Wales — coal
    "AU-SA":  250,   # South Australia — high wind/solar
    "DEFAULT": 400,  # Global average
}

REGION_AMPLITUDE: Dict[str, float] = {
    "US-CA":  80,    # Wide solar swing
    "EU-DE":  100,
    "DEFAULT": 60,
}


def _synthetic_intensity(region: str, dt: datetime) -> float:
    """Generate a realistic synthetic carbon intensity for a given time.

    Models a 24h sinusoidal cycle with:
      - Morning ramp-up (demand rises)
      - Midday peak (gas peakers running)
      - Solar dip in renewable-heavy regions
      - Evening peak
      - Night low
    """
    base = REGION_BASELINES.get(region, REGION_BASELINES["DEFAULT"])
    amp = REGION_AMPLITUDE.get(region, REGION_AMPLITUDE["DEFAULT"])

    # Hour of day in [0, 24)
    h = dt.hour + dt.minute / 60.0

    # Two-peak daily pattern (morning ~9h, evening ~19h) in radians
    # Shift so the minimum is around 3am
    phase_shift = 2 * math.pi * (3 / 24)
    diurnal = amp * math.sin(2 * math.pi * (h / 24) - phase_shift)

    # Secondary solar dip (midday) for renewable-heavy regions
    solar_amp = max(0, base - 150) / 5  # Only meaningful if base is moderate
    solar_dip = solar_amp * math.exp(-0.5 * ((h - 13) / 2) ** 2)  # Gaussian at 1pm

    # Day-of-week: weekends slightly lower demand (less industrial)
    dow_factor = 0.95 if dt.weekday() >= 5 else 1.0

    intensity = (base + diurnal - solar_dip) * dow_factor

    # Small stochastic noise ±3%
    noise = intensity * random.gauss(0, 0.015)

    return max(20, intensity + noise)  # Floor at 20 gCO2/kWh (near-zero carbon grid)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class CarbonIntensityProvider:
    """Provides carbon intensity readings for a given region.

    Args:
        region: Region code, e.g. "US-CA", "EU-DE". See REGION_BASELINES for
                recognized codes; unrecognized codes use a default signal.
        mode: "synthetic" (no API key, always works) or "electricitymaps"
              (requires ELECTRICITY_MAPS_API_KEY env var, falls back to
              synthetic on failure).
        api_key: Override for ElectricityMaps API key (default: env var).
        cache_ttl_seconds: How long to cache live readings before re-fetching.
        forecast_noise_factor: Noise multiplier for synthetic forecasts (0–1).
    """

    ELECTRICITYMAPS_URL = "https://api.electricitymap.org/v3/carbon-intensity/latest"
    ELECTRICITYMAPS_FORECAST_URL = "https://api.electricitymap.org/v3/carbon-intensity/forecast"

    def __init__(
        self,
        region: str = "US-CA",
        mode: str = "synthetic",
        api_key: Optional[str] = None,
        cache_ttl_seconds: int = 300,
        forecast_noise_factor: float = 0.05,
    ) -> None:
        self.region = region
        self.mode = mode
        self.api_key = api_key or os.environ.get("ELECTRICITY_MAPS_API_KEY")
        self.cache_ttl_seconds = cache_ttl_seconds
        self.forecast_noise_factor = forecast_noise_factor
        self._cache: Optional[CarbonReading] = None
        self._cache_ts: float = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def current(self) -> CarbonReading:
        """Return current carbon intensity."""
        now = time.monotonic()
        if self._cache and (now - self._cache_ts) < self.cache_ttl_seconds:
            return self._cache

        reading = self._fetch_current()
        self._cache = reading
        self._cache_ts = now
        return reading

    def forecast(
        self, hours: int = 48, resolution_minutes: int = 60
    ) -> List[CarbonReading]:
        """Return a forecast of carbon intensity over the next `hours` hours.

        Each data point is spaced `resolution_minutes` apart.

        In synthetic mode this is deterministic (+ small noise), so you get
        consistent planning results across calls.
        """
        if self.mode == "electricitymaps" and self.api_key:
            readings = self._fetch_em_forecast(hours)
            if readings:
                return readings

        # Fall back / default: synthetic forecast
        return self._synthetic_forecast(hours, resolution_minutes)

    def historical(self, hours_back: int = 24) -> List[CarbonReading]:
        """Return synthetic historical readings (for testing/demo)."""
        now = datetime.utcnow()
        readings = []
        for i in range(hours_back, 0, -1):
            dt = now - timedelta(hours=i)
            readings.append(CarbonReading(
                timestamp=dt,
                intensity_gco2_kwh=_synthetic_intensity(self.region, dt),
                region=self.region,
                source="synthetic",
                is_forecast=False,
            ))
        return readings

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch_current(self) -> CarbonReading:
        if self.mode == "electricitymaps" and self.api_key:
            reading = self._fetch_em_current()
            if reading:
                return reading
            logger.warning("ElectricityMaps fetch failed; using synthetic data")

        # Synthetic
        now = datetime.utcnow()
        return CarbonReading(
            timestamp=now,
            intensity_gco2_kwh=_synthetic_intensity(self.region, now),
            region=self.region,
            source="synthetic",
        )

    def _fetch_em_current(self) -> Optional[CarbonReading]:
        try:
            resp = requests.get(
                self.ELECTRICITYMAPS_URL,
                params={"zone": self.region},
                headers={"auth-token": self.api_key},
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()
            return CarbonReading(
                timestamp=datetime.utcnow(),
                intensity_gco2_kwh=float(data["carbonIntensity"]),
                region=self.region,
                source="electricitymaps",
            )
        except Exception as exc:
            logger.warning(f"ElectricityMaps API error: {exc}")
            return None

    def _fetch_em_forecast(self, hours: int) -> Optional[List[CarbonReading]]:
        try:
            resp = requests.get(
                self.ELECTRICITYMAPS_FORECAST_URL,
                params={"zone": self.region},
                headers={"auth-token": self.api_key},
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()
            readings = []
            now = datetime.utcnow()
            cutoff = now + timedelta(hours=hours)
            for point in data.get("forecast", []):
                ts = datetime.fromisoformat(point["datetime"].replace("Z", "+00:00")).replace(tzinfo=None)
                if ts > cutoff:
                    break
                readings.append(CarbonReading(
                    timestamp=ts,
                    intensity_gco2_kwh=float(point["carbonIntensity"]),
                    region=self.region,
                    source="electricitymaps",
                    is_forecast=True,
                ))
            return readings if readings else None
        except Exception as exc:
            logger.warning(f"ElectricityMaps forecast error: {exc}")
            return None

    def _synthetic_forecast(
        self, hours: int, resolution_minutes: int
    ) -> List[CarbonReading]:
        now = datetime.utcnow()
        readings = []
        steps = int(hours * 60 / resolution_minutes)
        for i in range(steps):
            dt = now + timedelta(minutes=i * resolution_minutes)
            base_intensity = _synthetic_intensity(self.region, dt)
            # Add small forecast uncertainty that grows with horizon
            horizon_hours = i * resolution_minutes / 60
            noise_scale = base_intensity * self.forecast_noise_factor * (1 + horizon_hours / 48)
            intensity = max(20, base_intensity + random.gauss(0, noise_scale))
            readings.append(CarbonReading(
                timestamp=dt,
                intensity_gco2_kwh=intensity,
                region=self.region,
                source="synthetic",
                is_forecast=True,
            ))
        return readings
