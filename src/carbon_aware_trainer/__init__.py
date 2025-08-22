"""
Carbon-Aware-Trainer: Intelligent ML training scheduler for carbon reduction.

This package provides tools to automatically schedule and optimize machine learning
training runs based on real-time carbon intensity forecasts, reducing the carbon
footprint of AI/ML workloads by 40-80%.
"""

# Core imports that should always work
from .core.scheduler import CarbonAwareTrainer
from .core.types import CarbonIntensity, CarbonForecast, TrainingConfig, TrainingMetrics

# Optional imports with graceful fallback
try:
    from .core.monitor import CarbonMonitor
except ImportError as e:
    print(f"Warning: Could not import CarbonMonitor: {e}")
    CarbonMonitor = None

try:
    from .core.forecasting import CarbonForecaster
except ImportError as e:
    print(f"Warning: Could not import CarbonForecaster: {e}")
    CarbonForecaster = None

try:
    from .core.multi_region import MultiRegionOrchestrator, PlacementPlan, RegionConfig
except ImportError as e:
    print(f"Warning: Could not import multi-region components: {e}")
    MultiRegionOrchestrator = PlacementPlan = RegionConfig = None

try:
    from .core.real_time_optimizer import RealTimeOptimizer, OptimizationMode, OptimizationAction
except ImportError as e:
    print(f"Warning: Could not import real-time optimizer: {e}")
    RealTimeOptimizer = OptimizationMode = OptimizationAction = None

try:
    from .core.federated_carbon import CarbonAwareFederated, FederatedClient, ClientSelection
except ImportError as e:
    print(f"Warning: Could not import federated components: {e}")
    CarbonAwareFederated = FederatedClient = ClientSelection = None

try:
    from .strategies.threshold import ThresholdScheduler
except ImportError as e:
    print(f"Warning: Could not import ThresholdScheduler: {e}")
    ThresholdScheduler = None

try:
    from .strategies.adaptive import AdaptiveScheduler
except ImportError as e:
    print(f"Warning: Could not import AdaptiveScheduler: {e}")
    AdaptiveScheduler = None

try:
    from .integrations.pytorch import CarbonAwarePyTorchTrainer
except ImportError as e:
    print(f"Warning: Could not import PyTorch integration: {e}")
    CarbonAwarePyTorchTrainer = None

try:
    from .integrations.lightning import CarbonAwareCallback, create_carbon_aware_callback
except ImportError as e:
    print(f"Warning: Could not import Lightning integration: {e}")
    CarbonAwareCallback = create_carbon_aware_callback = None

try:
    from .monitoring.metrics import MetricsCollector
except ImportError as e:
    print(f"Warning: Could not import MetricsCollector: {e}")
    MetricsCollector = None

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

__all__ = [
    # Core classes
    "CarbonAwareTrainer",
    "CarbonMonitor", 
    "CarbonForecaster",
    # Advanced orchestration
    "MultiRegionOrchestrator",
    "RealTimeOptimizer",
    "CarbonAwareFederated",
    # Data types
    "CarbonIntensity",
    "CarbonForecast", 
    "TrainingConfig",
    "TrainingMetrics",
    "PlacementPlan",
    "RegionConfig",
    "FederatedClient",
    "ClientSelection",
    "OptimizationMode",
    "OptimizationAction",
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