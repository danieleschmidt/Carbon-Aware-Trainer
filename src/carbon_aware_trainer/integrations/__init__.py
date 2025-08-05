"""Framework integrations for carbon-aware training."""

from .pytorch import CarbonAwarePyTorchTrainer
from .lightning import CarbonAwareCallback

__all__ = ["CarbonAwarePyTorchTrainer", "CarbonAwareCallback"]