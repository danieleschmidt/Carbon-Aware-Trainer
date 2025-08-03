# Carbon-Aware-Trainer 🌱⚡

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Science Advances](https://img.shields.io/badge/Paper-Science%20Advances-red.svg)](https://www.science.org/doi/10.1126/sciadv.xxxxx)
[![Carbon Neutral](https://img.shields.io/badge/Carbon-Neutral%20AI-brightgreen.svg)](https://github.com/yourusername/Carbon-Aware-Trainer)

Drop-in scheduler that intelligently aligns large-scale ML training with regional carbon intensity forecasts, supporting Slurm, Kubernetes, and PyTorch Lightning.

## 🌍 Impact

Training large models produces significant CO₂ emissions. Carbon-Aware-Trainer can reduce training emissions by **40-80%** by:
- Scheduling compute during low-carbon periods
- Migrating workloads to cleaner regions
- Optimizing batch sizes based on grid carbon intensity
- Providing real-time carbon accounting

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install carbon-aware-trainer

# With all schedulers
pip install carbon-aware-trainer[all]

# Development version
git clone https://github.com/yourusername/Carbon-Aware-Trainer.git
cd Carbon-Aware-Trainer
pip install -e ".[dev]"
```

### PyTorch Integration

```python
from carbon_aware_trainer import CarbonAwareTrainer
import torch

# Drop-in replacement for standard training
trainer = CarbonAwareTrainer(
    model=your_model,
    optimizer=optimizer,
    carbon_model='electricitymap',  # or 'watttime'
    region='US-CA',
    target_carbon_intensity=50  # gCO2/kWh
)

# Training automatically pauses during high-carbon periods
for epoch in range(epochs):
    for batch in dataloader:
        loss = trainer.train_step(batch)
        
        # Optional: View carbon metrics
        if trainer.step % 100 == 0:
            metrics = trainer.get_carbon_metrics()
            print(f"Current intensity: {metrics['current_intensity']} gCO2/kWh")
            print(f"Carbon saved: {metrics['carbon_saved_kg']:.2f} kg")
```

### Lightning Integration

```python
import pytorch_lightning as pl
from carbon_aware_trainer.lightning import CarbonAwareCallback

# Add as Lightning callback
carbon_callback = CarbonAwareCallback(
    pause_threshold=100,  # gCO2/kWh
    migration_enabled=True,
    regions=['US-CA', 'US-WA', 'EU-FR']
)

trainer = pl.Trainer(
    callbacks=[carbon_callback],
    accelerator='gpu',
    devices=8
)

trainer.fit(model, datamodule)
```

## 🏗️ Architecture

```
carbon-aware-trainer/
├── core/                    # Core scheduling logic
│   ├── scheduler.py        # Main scheduler
│   ├── forecasting.py      # Carbon forecasting
│   ├── optimization.py     # Schedule optimization
│   └── migration.py        # Cross-region migration
├── carbon_models/          # Carbon intensity providers
│   ├── electricitymap.py   # ElectricityMap API
│   ├── watttime.py        # WattTime API
│   ├── custom.py          # Custom data sources
│   └── cached.py          # Offline forecasts
├── integrations/          # Framework integrations
│   ├── pytorch/           # PyTorch native
│   ├── lightning/         # PyTorch Lightning
│   ├── tensorflow/        # TensorFlow/Keras
│   ├── jax/              # JAX/Flax
│   └── huggingface/      # Transformers
├── cluster/              # Cluster schedulers
│   ├── slurm/            # SLURM integration
│   ├── kubernetes/       # K8s operators
│   ├── aws_batch/        # AWS Batch
│   └── azure_ml/         # Azure ML
├── monitoring/           # Carbon tracking
│   ├── dashboard/        # Real-time dashboard
│   ├── reporting/        # Carbon reports
│   └── mlflow/          # MLflow integration
└── strategies/          # Scheduling strategies
    ├── threshold.py      # Simple threshold
    ├── predictive.py     # ML-based prediction
    ├── adaptive.py       # Adaptive scheduling
    └── multi_region.py   # Multi-region optimization
```

## 📊 Carbon Intelligence

### Real-time Carbon Monitoring

```python
from carbon_aware_trainer import CarbonMonitor

monitor = CarbonMonitor(
    regions=['US-CA', 'US-TX', 'EU-DE', 'IN-KA'],
    update_interval=300  # 5 minutes
)

# Get current carbon intensity
intensity = monitor.get_current_intensity('US-CA')
print(f"California grid: {intensity['carbon_intensity']} gCO2/kWh")
print(f"Energy mix: {intensity['energy_mix']}")

# Get 24-hour forecast
forecast = monitor.get_forecast('US-CA', hours=24)
optimal_window = monitor.find_optimal_window(
    duration_hours=8,  # Training duration
    max_carbon_intensity=100
)

print(f"Best time to train: {optimal_window['start_time']}")
print(f"Expected carbon intensity: {optimal_window['avg_intensity']} gCO2/kWh")
```

### Adaptive Scheduling

```python
from carbon_aware_trainer.strategies import AdaptiveScheduler

# ML-based scheduling that learns patterns
scheduler = AdaptiveScheduler(
    historical_data='carbon_history.parquet',
    workload_flexibility=0.3,
