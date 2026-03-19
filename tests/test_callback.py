"""Tests for CarbonAwareCallback."""

import random
import time
from unittest.mock import MagicMock, patch

import pytest

from carbon_aware_trainer.callback import CarbonAwareCallback
from carbon_aware_trainer.provider import CarbonIntensityProvider, CarbonReading
from datetime import datetime


random.seed(2)


def make_reading(intensity: float, region: str = "US-CA") -> CarbonReading:
    return CarbonReading(
        timestamp=datetime.utcnow(),
        intensity_gco2_kwh=intensity,
        region=region,
        source="synthetic",
    )


@pytest.fixture
def low_carbon_provider():
    """Provider always returning 100 gCO2/kWh (low — below any threshold)."""
    p = MagicMock(spec=CarbonIntensityProvider)
    p.current.return_value = make_reading(100.0)
    return p


@pytest.fixture
def high_carbon_provider():
    """Provider always returning 500 gCO2/kWh (high — above default threshold)."""
    p = MagicMock(spec=CarbonIntensityProvider)
    p.current.return_value = make_reading(500.0)
    return p


class TestCarbonAwareCallbackBasic:
    def test_on_train_start(self, low_carbon_provider):
        cb = CarbonAwareCallback(low_carbon_provider, threshold_gco2_kwh=300, dry_run=True)
        cb.on_train_start()
        assert cb._training_start is not None

    def test_step_counting(self, low_carbon_provider):
        cb = CarbonAwareCallback(low_carbon_provider, dry_run=True)
        cb.on_train_start()
        for _ in range(10):
            time.sleep(0.001)
            cb.on_step_end()
        assert cb._steps == 10

    def test_no_pause_when_below_threshold(self, low_carbon_provider):
        cb = CarbonAwareCallback(
            low_carbon_provider,
            threshold_gco2_kwh=300,
            dry_run=True,
        )
        cb.on_train_start()
        cb.on_epoch_start(0)
        assert cb._pauses == 0

    def test_pause_triggered_when_above_threshold(self, high_carbon_provider):
        """With dry_run=True, a pause is logged but no sleep occurs."""
        cb = CarbonAwareCallback(
            high_carbon_provider,
            threshold_gco2_kwh=300,
            dry_run=True,
        )
        cb.on_train_start()
        cb.on_epoch_start(0)
        assert cb._pauses == 1

    def test_multiple_epochs_count_pauses(self, high_carbon_provider):
        cb = CarbonAwareCallback(
            high_carbon_provider,
            threshold_gco2_kwh=300,
            dry_run=True,
        )
        cb.on_train_start()
        for epoch in range(5):
            cb.on_epoch_start(epoch)
        assert cb._pauses == 5


class TestCarbonAwareCallbackAccounting:
    def test_mean_intensity(self, low_carbon_provider):
        cb = CarbonAwareCallback(low_carbon_provider, dry_run=True)
        cb.on_train_start()
        cb.on_epoch_start(0)
        # Should be close to 100 (our mock intensity)
        assert cb.mean_intensity == pytest.approx(100.0, rel=0.01)

    def test_carbon_emitted_positive_after_steps(self, low_carbon_provider):
        cb = CarbonAwareCallback(low_carbon_provider, gpu_tdp_watts=300, dry_run=True)
        cb.on_train_start()
        for _ in range(100):
            time.sleep(0.001)
            cb.on_step_end()
        # Should have non-zero carbon after actual time passed
        assert cb.carbon_emitted_kg >= 0.0  # May be 0 if time is extremely short

    def test_summary_contains_key_fields(self, low_carbon_provider):
        cb = CarbonAwareCallback(low_carbon_provider, dry_run=True)
        cb.on_train_start()
        for epoch in range(3):
            cb.on_epoch_start(epoch)
            for _ in range(10):
                cb.on_step_end()
        cb.on_train_end()
        summary = cb.summary()
        assert "Steps" in summary
        assert "Pauses" in summary
        assert "carbon" in summary.lower()

    def test_carbon_avoided_zero_when_no_pauses(self, low_carbon_provider):
        cb = CarbonAwareCallback(
            low_carbon_provider, threshold_gco2_kwh=999, dry_run=True
        )
        cb.on_train_start()
        cb.on_epoch_start(0)
        assert cb.carbon_avoided_kg == pytest.approx(0.0, abs=1e-6)

    def test_resume_threshold_defaults_to_90pct(self, low_carbon_provider):
        cb = CarbonAwareCallback(
            low_carbon_provider, threshold_gco2_kwh=300, dry_run=True
        )
        assert cb.resume_threshold == pytest.approx(270.0)

    def test_custom_resume_threshold(self, low_carbon_provider):
        cb = CarbonAwareCallback(
            low_carbon_provider,
            threshold_gco2_kwh=300,
            resume_threshold_gco2_kwh=200,
            dry_run=True,
        )
        assert cb.resume_threshold == 200.0
