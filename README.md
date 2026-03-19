# Carbon-Aware Trainer 🌱

> Shift your ML training jobs to low-carbon time windows and cut your CO2 footprint without changing your model code.

## The idea

The carbon intensity of the electrical grid varies throughout the day — sometimes dramatically. In California, solar floods the grid at midday, dropping intensity below 100 gCO2/kWh. In Texas, evening peakers push it past 500. A 6-hour training run on a 300W A100 can emit 280g or 130g CO2 depending purely on *when* you start it.

This library does one thing well: **find the greenest time window for your job and stay there**.

```
Naive start:   2026-03-19 15:00 UTC  |  157 gCO2/kWh  |  282g CO2
Optimal start: 2026-03-19 19:00 UTC  |  129 gCO2/kWh  |  231g CO2
                                                 saves  ▶  51g CO2  (18%)
```

Scale that up to a week-long fine-tune and you're talking kilograms.

---

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.8. No API key needed for synthetic mode (great for testing and demos).

For live grid data, get a free key at [ElectricityMaps](https://www.electricitymaps.com/) and set:
```bash
export ELECTRICITY_MAPS_API_KEY=your_key_here
```

---

## Quick start

### 1. Find the optimal window for a training job

```python
from carbon_aware_trainer import CarbonIntensityProvider, TrainingScheduler, TrainingJob

provider = CarbonIntensityProvider(region="US-CA")   # synthetic by default
scheduler = TrainingScheduler(provider, lookahead_hours=48)

job = TrainingJob(job_hours=6.0, gpu_tdp_watts=300.0, name="ResNet-50")
window = scheduler.find_optimal_window(job=job)

print(window.summary())
# Optimal start: 2026-03-19 19:00 UTC  (+4.0h delay)  |  saves 51g CO2 (18.0%)
```

### 2. Add carbon awareness to your training loop

```python
from carbon_aware_trainer import CarbonAwareCallback

cb = CarbonAwareCallback(
    provider,
    threshold_gco2_kwh=250,    # pause above this
    gpu_tdp_watts=300,
)

cb.on_train_start()
for epoch in range(num_epochs):
    cb.on_epoch_start(epoch)   # pauses here if grid is dirty
    for batch in dataloader:
        loss = train_step(batch)
        cb.on_step_end()

cb.on_train_end()
print(cb.summary())
```

### 3. Estimate carbon footprint

```python
from carbon_aware_trainer import EnergyEstimator

est = EnergyEstimator.full_estimate(
    gpu_tdp_watts=400,   # A100
    hours=6,
    intensity_gco2_kwh=200,
)
print(f"Energy: {est['energy_kwh']} kWh  |  CO2: {est['carbon_g']}g")
# Energy: 2.4 kWh  |  CO2: 480.0g
```

---

## Components

### `CarbonIntensityProvider`

Provides carbon intensity data (gCO2/kWh) for a grid region.

| Mode | Description |
|------|-------------|
| `synthetic` | Sinusoidal 24h model based on typical grid patterns. No API key. Works offline. |
| `electricitymaps` | Live + forecast data from [ElectricityMaps API](https://www.electricitymaps.com/). Falls back to synthetic on failure. |

**Supported regions (synthetic):** `US-CA`, `US-TX`, `US-NY`, `US-MW`, `US-NE`, `EU-DE`, `EU-FR`, `EU-GB`, `EU-PL`, `AU-NSW`, `AU-SA` — plus any ElectricityMaps zone code in live mode.

```python
provider = CarbonIntensityProvider(
    region="US-CA",
    mode="synthetic",           # or "electricitymaps"
    api_key=None,               # or ELECTRICITY_MAPS_API_KEY env var
    cache_ttl_seconds=300,
)

reading = provider.current()       # CarbonReading right now
forecast = provider.forecast(hours=48)  # List[CarbonReading]
```

### `TrainingScheduler`

Slides a job-duration window across the forecast to find the minimum-carbon start time.

```python
scheduler = TrainingScheduler(provider, lookahead_hours=48, resolution_minutes=60)
window = scheduler.find_optimal_window(job_hours=6.0, gpu_tdp_watts=300)

window.start_time                    # datetime
window.mean_intensity_gco2_kwh       # float
window.carbon_kg                     # float
window.carbon_saved_kg               # vs naive (start now)
window.carbon_saved_pct              # percentage savings
```

### `CarbonAwareCallback`

Drop-in callback that pauses training when grid intensity exceeds a threshold and resumes when it drops back (with hysteresis).

```python
cb = CarbonAwareCallback(
    provider,
    threshold_gco2_kwh=300,         # pause above this
    resume_threshold_gco2_kwh=270,  # resume below this (default: 90% of threshold)
    gpu_tdp_watts=300,
    check_interval_seconds=60,      # how often to re-check while paused
    max_pause_seconds=0,            # 0 = no time limit on pauses
    dry_run=False,                  # True = log only, never sleep
)
```

Hooks: `on_train_start()`, `on_epoch_start(epoch)`, `on_step_end()`, `on_train_end()`

### `EnergyEstimator`

Simple carbon math — no dependencies.

```python
EnergyEstimator.energy_kwh(gpu_tdp_watts=300, hours=6)       # → 1.8 kWh
EnergyEstimator.carbon_kg(energy_kwh=1.8, intensity_gco2_kwh=200)  # → 0.36 kg
EnergyEstimator.full_estimate(300, 6, 200)                   # → dict
EnergyEstimator.tdp_for_gpu("A100")                          # → 400
```

---

## Demo

```bash
python demo.py
```

Shows:
1. 48-hour carbon intensity forecast with ASCII sparkline
2. Optimal vs. naive scheduling for a 6-hour job (CO2 savings)
3. GPU comparison table (V100, A100, H100, RTX 4090)
4. CarbonAwareCallback dry-run with epoch-based pause simulation

---

## Tests

```bash
pytest tests/ -v
```

50 tests, no external dependencies needed.

---

## Why this matters

A single A100 training run that shifts 4 hours saves ~50g CO2. At scale:

| Scenario | Annual CO2 saved |
|----------|-----------------|
| 1 researcher, 1 job/day | ~18 kg/year |
| 10-GPU cluster, continuous | ~1.8 tonnes/year |
| 100-GPU cluster | ~18 tonnes/year |

The biggest levers: **region** (France's nuclear grid vs Australia's coal) and **timing** (solar surplus windows). This library handles the timing piece automatically.

---

## License

MIT
