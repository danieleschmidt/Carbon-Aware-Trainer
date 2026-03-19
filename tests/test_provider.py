"""Tests for CarbonIntensityProvider."""

import random
from datetime import datetime, timedelta

import pytest

from carbon_aware_trainer.provider import (
    CarbonIntensityProvider,
    CarbonReading,
    _synthetic_intensity,
    REGION_BASELINES,
)


random.seed(0)


class TestSyntheticIntensity:
    def test_returns_positive_value(self):
        dt = datetime(2024, 6, 15, 12, 0)
        result = _synthetic_intensity("US-CA", dt)
        assert result > 0

    def test_never_below_floor(self):
        """Intensity should never go below 20 gCO2/kWh."""
        for hour in range(24):
            dt = datetime(2024, 6, 15, hour, 0)
            assert _synthetic_intensity("US-CA", dt) >= 20

    def test_night_lower_than_peak(self):
        """Night (3am) should generally be lower than midday peak."""
        night = _synthetic_intensity("US-CA", datetime(2024, 6, 15, 3, 0))
        # Average over multiple midday points to smooth noise
        midday_values = [
            _synthetic_intensity("US-CA", datetime(2024, 6, 15, 12, m))
            for m in range(0, 60, 5)
        ]
        avg_midday = sum(midday_values) / len(midday_values)
        # Night should be below average midday (not necessarily every single reading)
        assert night < avg_midday

    def test_unknown_region_uses_default(self):
        dt = datetime(2024, 6, 15, 12, 0)
        result = _synthetic_intensity("XX-UNKNOWN", dt)
        assert result > 0

    def test_known_regions_have_reasonable_values(self):
        dt = datetime(2024, 6, 15, 12, 0)
        for region in ["US-CA", "EU-FR", "AU-NSW"]:
            val = _synthetic_intensity(region, dt)
            assert 20 <= val <= 1000, f"Region {region} out of range: {val}"


class TestCarbonReading:
    def test_repr(self):
        r = CarbonReading(
            timestamp=datetime(2024, 6, 15, 14, 30),
            intensity_gco2_kwh=250.5,
            region="US-CA",
        )
        assert "US-CA" in repr(r)
        assert "250" in repr(r)
        assert "14:30" in repr(r)


class TestCarbonIntensityProvider:
    def test_current_returns_reading(self):
        p = CarbonIntensityProvider(region="US-CA", mode="synthetic")
        reading = p.current()
        assert isinstance(reading, CarbonReading)
        assert reading.region == "US-CA"
        assert reading.intensity_gco2_kwh > 0
        assert reading.source == "synthetic"

    def test_current_is_cached(self):
        p = CarbonIntensityProvider(region="US-CA", mode="synthetic", cache_ttl_seconds=60)
        r1 = p.current()
        r2 = p.current()
        assert r1 is r2  # Same object from cache

    def test_forecast_length(self):
        p = CarbonIntensityProvider(region="US-CA", mode="synthetic")
        forecast = p.forecast(hours=24, resolution_minutes=60)
        assert len(forecast) == 24

    def test_forecast_is_future(self):
        p = CarbonIntensityProvider(region="US-CA", mode="synthetic")
        forecast = p.forecast(hours=6, resolution_minutes=60)
        now = datetime.utcnow()
        for r in forecast:
            assert r.timestamp > now - timedelta(minutes=5)  # Allow small offset

    def test_forecast_marked_as_forecast(self):
        p = CarbonIntensityProvider(region="US-CA", mode="synthetic")
        forecast = p.forecast(hours=6, resolution_minutes=60)
        for r in forecast:
            assert r.is_forecast is True

    def test_forecast_resolution_60_min(self):
        p = CarbonIntensityProvider(region="US-CA", mode="synthetic")
        forecast = p.forecast(hours=4, resolution_minutes=60)
        assert len(forecast) == 4
        gaps = [(forecast[i+1].timestamp - forecast[i].timestamp).total_seconds()
                for i in range(len(forecast) - 1)]
        for gap in gaps:
            assert abs(gap - 3600) < 10, f"Expected 60min gap, got {gap}s"

    def test_forecast_resolution_30_min(self):
        p = CarbonIntensityProvider(region="US-CA", mode="synthetic")
        forecast = p.forecast(hours=4, resolution_minutes=30)
        assert len(forecast) == 8

    def test_historical_returns_past(self):
        p = CarbonIntensityProvider(region="EU-FR", mode="synthetic")
        hist = p.historical(hours_back=6)
        assert len(hist) == 6
        now = datetime.utcnow()
        for r in hist:
            assert r.timestamp < now

    def test_electricitymaps_falls_back_to_synthetic(self):
        """Without an API key, electricitymaps mode should fall back."""
        p = CarbonIntensityProvider(region="US-CA", mode="electricitymaps", api_key=None)
        reading = p.current()
        # Should still return a reading (synthetic fallback)
        assert reading.intensity_gco2_kwh > 0
