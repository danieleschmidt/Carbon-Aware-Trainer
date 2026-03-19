"""Energy and carbon estimation utilities.

Simple formulas — no external dependencies.

  energy_kwh = gpu_tdp_watts * job_hours / 1000
  carbon_kg  = energy_kwh * intensity_gco2_kwh / 1000

Reference GPU TDP values (approximate):
  RTX 3090   : 350W
  RTX 4090   : 450W
  A100 SXM4  : 400W  (PCIe variant: 300W)
  H100 SXM5  : 700W
  V100       : 300W
"""

from __future__ import annotations


# Approximate TDP values in watts for common training GPUs
GPU_TDP: dict[str, float] = {
    "V100": 300,
    "A100": 400,
    "A100-PCIe": 300,
    "H100": 700,
    "H100-PCIe": 350,
    "RTX3090": 350,
    "RTX4090": 450,
    "RTX3080": 320,
    "RTX4080": 320,
    "A6000": 300,
    "A40": 300,
}


class EnergyEstimator:
    """Static helpers for energy and carbon footprint calculations."""

    @staticmethod
    def energy_kwh(gpu_tdp_watts: float, hours: float) -> float:
        """Compute energy in kWh from GPU TDP and training duration.

        Args:
            gpu_tdp_watts: GPU TDP in watts.
            hours: Training duration in hours.

        Returns:
            Energy in kWh.
        """
        return (gpu_tdp_watts * hours) / 1000.0

    @staticmethod
    def carbon_kg(energy_kwh: float, intensity_gco2_kwh: float) -> float:
        """Compute carbon emissions in kg CO2.

        Args:
            energy_kwh: Energy consumed in kWh.
            intensity_gco2_kwh: Grid carbon intensity in gCO2/kWh.

        Returns:
            Carbon emissions in kg CO2.
        """
        return (energy_kwh * intensity_gco2_kwh) / 1000.0

    @staticmethod
    def full_estimate(
        gpu_tdp_watts: float,
        hours: float,
        intensity_gco2_kwh: float,
    ) -> dict:
        """Full estimation: energy + carbon from GPU specs and grid intensity.

        Args:
            gpu_tdp_watts: GPU TDP in watts.
            hours: Training duration in hours.
            intensity_gco2_kwh: Grid carbon intensity in gCO2/kWh.

        Returns:
            Dict with energy_kwh, carbon_kg, carbon_g.
        """
        energy_kwh = EnergyEstimator.energy_kwh(gpu_tdp_watts, hours)
        carbon_kg = EnergyEstimator.carbon_kg(energy_kwh, intensity_gco2_kwh)
        return {
            "gpu_tdp_watts": gpu_tdp_watts,
            "training_hours": hours,
            "energy_kwh": round(energy_kwh, 4),
            "intensity_gco2_kwh": intensity_gco2_kwh,
            "carbon_kg": round(carbon_kg, 4),
            "carbon_g": round(carbon_kg * 1000, 2),
        }

    @staticmethod
    def tdp_for_gpu(gpu_name: str) -> float:
        """Look up TDP for a named GPU. Raises KeyError if unknown."""
        key = gpu_name.upper().replace(" ", "").replace("-", "")
        for k, v in GPU_TDP.items():
            if k.upper().replace("-", "") == key:
                return v
        raise KeyError(
            f"Unknown GPU '{gpu_name}'. Known GPUs: {list(GPU_TDP.keys())}"
        )
