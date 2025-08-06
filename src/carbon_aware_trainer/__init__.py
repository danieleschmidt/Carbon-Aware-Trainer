"""
Carbon-Aware-Trainer: Intelligent ML training scheduler for carbon reduction.

This package provides tools to automatically schedule and optimize machine learning
training runs based on real-time carbon intensity forecasts, reducing the carbon
footprint of AI/ML workloads by 40-80%.
"""

from .core.scheduler import CarbonAwareTrainer
from .core.monitor import CarbonMonitor
from .core.forecasting import CarbonForecaster
from .core.types import CarbonIntensity, CarbonForecast, TrainingConfig, TrainingMetrics
from .strategies.threshold import ThresholdScheduler
from .strategies.adaptive import AdaptiveScheduler
from .integrations.pytorch import CarbonAwarePyTorchTrainer
from .integrations.lightning import CarbonAwareCallback, create_carbon_aware_callback
from .monitoring.metrics import MetricsCollector

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

__all__ = [
    # Core classes
    "CarbonAwareTrainer",
    "CarbonMonitor", 
    "CarbonForecaster",
    # Data types
    "CarbonIntensity",
    "CarbonForecast", 
    "TrainingConfig",
    "TrainingMetrics",
    # Strategies
    "ThresholdScheduler",
    "AdaptiveScheduler",
    # Integrations
    "CarbonAwarePyTorchTrainer",
    "CarbonAwareCallback",
    "create_carbon_aware_callback",
    # Monitoring
    "MetricsCollector",
]