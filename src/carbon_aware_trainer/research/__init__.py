"""Research-oriented features for carbon-aware training."""

from .experimental_framework import ExperimentalFramework, ExperimentConfig
from .baseline_comparator import BaselineComparator
from .statistical_analyzer import StatisticalAnalyzer
from .reproducibility import ReproducibilityManager

__all__ = [
    "ExperimentalFramework",
    "ExperimentConfig", 
    "BaselineComparator",
    "StatisticalAnalyzer",
    "ReproducibilityManager"
]