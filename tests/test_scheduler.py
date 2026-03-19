"""Tests for TrainingScheduler."""

import random
from datetime import datetime

import pytest

from carbon_aware_trainer.provider import CarbonIntensityProvider
from carbon_aware_trainer.scheduler import TrainingScheduler, TrainingJob, ScheduledWindow


random.seed(1)


@pytest.fixture
def provider():
    return CarbonIntensityProvider(region="US-CA", mode="synthetic")


@pytest.fixture
def scheduler(provider):
    return TrainingScheduler(provider, lookahead_hours=48, resolution_minutes=60)


class TestTrainingJob:
    def test_energy_kwh(self):
        job = TrainingJob(job_hours=6.0, gpu_tdp_watts=300.0)
        assert job.energy_kwh == pytest.approx(1.8)

    def test_name_default(self):
        job = TrainingJob(job_hours=2.0)
        assert job.name == "training-job"


class TestTrainingScheduler:
    def test_find_optimal_window_returns_window(self, scheduler):
        window = scheduler.find_optimal_window(job_hours=6.0)
        assert isinstance(window, ScheduledWindow)

    def test_optimal_window_intensity_le_naive(self, scheduler):
        """Optimal window should have ≤ mean intensity than starting now."""
        window = scheduler.find_optimal_window(job_hours=6.0)
        assert window.mean_intensity_gco2_kwh <= window.naive_mean_intensity_gco2_kwh

    def test_carbon_saved_nonnegative(self, scheduler):
        window = scheduler.find_optimal_window(job_hours=6.0)
        assert window.carbon_saved_kg >= 0.0

    def test_carbon_saved_pct_in_range(self, scheduler):
        window = scheduler.find_optimal_window(job_hours=6.0)
        assert 0 <= window.carbon_saved_pct <= 100

    def test_end_time_after_start(self, scheduler):
        window = scheduler.find_optimal_window(job_hours=6.0)
        assert window.end_time > window.start_time

    def test_window_duration(self, scheduler):
        hours = 4.0
        window = scheduler.find_optimal_window(job_hours=hours)
        actual_h = (window.end_time - window.start_time).total_seconds() / 3600
        assert actual_h == pytest.approx(hours)

    def test_accepts_training_job_object(self, scheduler):
        job = TrainingJob(job_hours=8.0, gpu_tdp_watts=400.0, name="test-job")
        window = scheduler.find_optimal_window(job=job)
        assert isinstance(window, ScheduledWindow)

    def test_requires_job_or_hours(self, scheduler):
        with pytest.raises(ValueError):
            scheduler.find_optimal_window()

    def test_forecast_points_populated(self, scheduler):
        window = scheduler.find_optimal_window(job_hours=6.0)
        assert len(window.forecast_points) > 0

    def test_summary_string(self, scheduler):
        window = scheduler.find_optimal_window(job_hours=6.0)
        s = window.summary()
        assert "Optimal start" in s
        assert "saves" in s

    def test_hourly_carbon_profile(self, scheduler):
        profile = scheduler.hourly_carbon_profile(hours=12)
        assert len(profile) == 12
        for entry in profile:
            assert "time" in entry
            assert "intensity_gco2_kwh" in entry
            assert entry["intensity_gco2_kwh"] > 0

    def test_short_job(self, scheduler):
        window = scheduler.find_optimal_window(job_hours=1.0)
        assert window.mean_intensity_gco2_kwh > 0

    def test_job_longer_than_lookahead_is_clamped(self, provider):
        # 5h lookahead, 24h job — should clamp gracefully
        sched = TrainingScheduler(provider, lookahead_hours=5)
        window = sched.find_optimal_window(job_hours=24.0)
        assert isinstance(window, ScheduledWindow)
