#!/usr/bin/env python3
"""Carbon-Aware Trainer — demo script.

Demonstrates:
  1. CarbonIntensityProvider: synthetic 48h forecast for US-CA
  2. TrainingScheduler: find optimal window for a 6-hour training job
  3. EnergyEstimator: energy and carbon calculation
  4. CarbonAwareCallback: dry-run simulation of epoch-based pause/resume

Run:
    ~/anaconda3/bin/python3 demo.py
"""

import random
import sys
import textwrap
from datetime import datetime

# Seed for reproducibility in demo output
random.seed(42)

from carbon_aware_trainer import (
    CarbonIntensityProvider,
    TrainingScheduler,
    TrainingJob,
    CarbonAwareCallback,
    EnergyEstimator,
)


def separator(title: str = "") -> None:
    width = 70
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─' * pad} {title} {'─' * (width - pad - len(title) - 2)}")
    else:
        print("─" * width)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Carbon intensity forecast
# ─────────────────────────────────────────────────────────────────────────────

separator("1. Carbon Intensity Forecast — US-CA (next 48 hours)")

provider = CarbonIntensityProvider(region="US-CA", mode="synthetic")
current = provider.current()
print(f"Current reading : {current}")

forecast = provider.forecast(hours=48, resolution_minutes=60)
print(f"Forecast points : {len(forecast)} hourly readings")

# Show a mini-sparkline
low  = min(r.intensity_gco2_kwh for r in forecast)
high = max(r.intensity_gco2_kwh for r in forecast)
print(f"Range           : {low:.0f} – {high:.0f} gCO2/kWh")

# Print 12-hour sample table
print(f"\n{'Hour':>5}  {'Time (UTC)':>16}  {'gCO2/kWh':>10}  {'Bar'}")
print("─" * 55)
for r in forecast[:48:4]:   # Every 4 hours
    bar_len = int((r.intensity_gco2_kwh - 50) / 10)
    bar = "█" * max(0, bar_len)
    print(f"{forecast.index(r):>5}  {r.timestamp.strftime('%a %H:%M'):>16}  "
          f"{r.intensity_gco2_kwh:>10.0f}  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Schedule a 6-hour training job
# ─────────────────────────────────────────────────────────────────────────────

separator("2. Optimal Scheduling — 6-hour job, A100 GPU (300W TDP)")

scheduler = TrainingScheduler(provider, lookahead_hours=48, resolution_minutes=60)
job = TrainingJob(job_hours=6.0, gpu_tdp_watts=300.0, name="ResNet-50 fine-tune")

window = scheduler.find_optimal_window(job=job)

print(f"Job             : {job.name}")
print(f"Duration        : {job.job_hours}h  |  Energy: {job.energy_kwh:.2f} kWh")
print()
print(f"{'':4}  {'Start time':>24}  {'Mean intensity':>16}  {'CO2':>10}")
print(f"{'Naive':4}  {window.naive_start_time.strftime('%Y-%m-%d %H:%M UTC'):>24}  "
      f"{window.naive_mean_intensity_gco2_kwh:>14.0f}  "
      f"{window.naive_carbon_kg*1000:>8.1f}g")
print(f"{'Best':4}  {window.start_time.strftime('%Y-%m-%d %H:%M UTC'):>24}  "
      f"{window.mean_intensity_gco2_kwh:>14.0f}  "
      f"{window.carbon_kg*1000:>8.1f}g")
print()
print(f"Carbon savings  : {window.carbon_saved_kg*1000:.1f}g CO2 "
      f"({window.carbon_saved_pct:.1f}% reduction)")

delay_h = (window.start_time - window.naive_start_time).total_seconds() / 3600
print(f"Start delay     : {delay_h:.1f}h")
print()
print(f"One-line summary:")
print(f"  {window.summary()}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Energy estimator
# ─────────────────────────────────────────────────────────────────────────────

separator("3. Energy & Carbon Estimation — GPU comparison")

scenarios = [
    ("V100  (300W)", 300, 6, 400),    # High-intensity region
    ("A100  (400W)", 400, 6, 200),    # Low-intensity region
    ("H100  (700W)", 700, 6, 55),     # French nuclear grid
    ("RTX4090 (450W)", 450, 6, 750),  # Australian coal
]

print(f"{'GPU':>15}  {'Hours':>5}  {'Energy (kWh)':>13}  "
      f"{'Grid (gCO2/kWh)':>16}  {'CO2':>8}")
print("─" * 65)
for name, tdp, hours, intensity in scenarios:
    est = EnergyEstimator.full_estimate(tdp, hours, intensity)
    print(f"{name:>15}  {hours:>5}  {est['energy_kwh']:>13.2f}  "
          f"{intensity:>16}  {est['carbon_g']:>6.0f}g")


# ─────────────────────────────────────────────────────────────────────────────
# 4. CarbonAwareCallback dry-run simulation
# ─────────────────────────────────────────────────────────────────────────────

separator("4. CarbonAwareCallback — dry-run epoch simulation")

# Use a provider with a higher baseline to force some "pause" triggers in the demo
demo_provider = CarbonIntensityProvider(region="US-TX", mode="synthetic")

cb = CarbonAwareCallback(
    provider=demo_provider,
    threshold_gco2_kwh=400.0,
    gpu_tdp_watts=300.0,
    dry_run=True,          # Never actually sleep — just log
)

cb.on_train_start()

num_epochs = 12
num_steps_per_epoch = 100
step_time_seconds = 0.001  # Fast mock steps

import time

for epoch in range(num_epochs):
    cb.on_epoch_start(epoch)
    for _ in range(num_steps_per_epoch):
        time.sleep(step_time_seconds)
        cb.on_step_end()

cb.on_train_end()

print()
print(cb.summary())


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

separator("Summary")

print(textwrap.dedent(f"""
    Carbon-Aware Trainer can reduce ML training emissions by shifting jobs to
    low-carbon grid windows. For this demo job (6h, A100, US-CA):

      • Naive start  : {window.naive_carbon_kg*1000:.1f}g CO2
      • Optimal start: {window.carbon_kg*1000:.1f}g CO2
      • Savings      : {window.carbon_saved_kg*1000:.1f}g CO2 ({window.carbon_saved_pct:.1f}%)

    Scale this up: a week-long GPT fine-tune can save kilos of CO2 by waiting
    a few hours for clean grid conditions.

    Next steps:
      - Set ELECTRICITY_MAPS_API_KEY for live grid data
      - Use CarbonAwareCallback in your training loop
      - Run: carbon-trainer schedule --region US-CA --hours 6
""").strip())
