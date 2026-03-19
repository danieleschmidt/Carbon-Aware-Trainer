"""Tests for EnergyEstimator."""

import pytest
from carbon_aware_trainer.energy import EnergyEstimator, GPU_TDP


class TestEnergyEstimator:
    def test_energy_kwh_basic(self):
        # 300W GPU for 1 hour = 0.3 kWh
        assert EnergyEstimator.energy_kwh(300, 1.0) == pytest.approx(0.3)

    def test_energy_kwh_zero_hours(self):
        assert EnergyEstimator.energy_kwh(400, 0) == 0.0

    def test_carbon_kg_basic(self):
        # 1 kWh at 1000 gCO2/kWh = 1 kg
        assert EnergyEstimator.carbon_kg(1.0, 1000.0) == pytest.approx(1.0)

    def test_carbon_kg_zero(self):
        assert EnergyEstimator.carbon_kg(0, 500) == 0.0
        assert EnergyEstimator.carbon_kg(1, 0) == 0.0

    def test_full_estimate_keys(self):
        est = EnergyEstimator.full_estimate(300, 6, 200)
        assert "energy_kwh" in est
        assert "carbon_kg" in est
        assert "carbon_g" in est
        assert "gpu_tdp_watts" in est

    def test_full_estimate_values(self):
        # 300W × 6h = 1.8 kWh; 1.8 × 200 / 1000 = 0.36 kg CO2 = 360 g
        est = EnergyEstimator.full_estimate(300, 6, 200)
        assert est["energy_kwh"] == pytest.approx(1.8)
        assert est["carbon_kg"] == pytest.approx(0.36)
        assert est["carbon_g"] == pytest.approx(360.0)

    def test_tdp_for_known_gpu(self):
        assert EnergyEstimator.tdp_for_gpu("V100") == 300
        assert EnergyEstimator.tdp_for_gpu("A100") == 400
        assert EnergyEstimator.tdp_for_gpu("H100") == 700

    def test_tdp_unknown_gpu_raises(self):
        with pytest.raises(KeyError):
            EnergyEstimator.tdp_for_gpu("FakeGPU9000")

    def test_real_world_sanity(self):
        """GPT-3 training is ~500 MWh. Single 6h job on A100 should be much less."""
        est = EnergyEstimator.full_estimate(400, 6, 300)
        assert est["energy_kwh"] < 10  # Should be ~2.4 kWh
        assert est["carbon_g"] < 2000  # Should be ~720 g
