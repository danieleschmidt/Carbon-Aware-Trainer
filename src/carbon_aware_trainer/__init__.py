"""
Carbon-Aware-Trainer: Intelligent ML training scheduler for carbon reduction.

This package provides tools to automatically schedule and optimize machine learning
training runs based on real-time carbon intensity forecasts, reducing the carbon
footprint of AI/ML workloads by 40-80%.
"""

from .core.scheduler import CarbonAwareTrainer
from .core.monitor import CarbonMonitor
from .core.forecasting import CarbonForecaster
from .strategies.threshold import ThresholdScheduler
from .strategies.adaptive import AdaptiveScheduler

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

__all__ = [
    "CarbonAwareTrainer",
    "CarbonMonitor", 
    "CarbonForecaster",
    "ThresholdScheduler",
    "AdaptiveScheduler",
]