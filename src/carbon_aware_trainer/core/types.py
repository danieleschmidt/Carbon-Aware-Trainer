"""Core type definitions for carbon-aware training."""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field

try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    # Fallback BaseModel for when pydantic is not available
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    def Field(default=None, **kwargs):
        return default


class CarbonIntensityUnit(str, Enum):
    """Units for carbon intensity measurement."""
    GRAMS_CO2_PER_KWH = "gCO2/kWh"
    KG_CO2_PER_MWH = "kgCO2/MWh"
    LBS_CO2_PER_MWH = "lbsCO2/MWh"


class TrainingState(str, Enum):
    """Current state of carbon-aware training."""
    RUNNING = "running"
    PAUSED = "paused"
    WAITING = "waiting"
    MIGRATING = "migrating"
    STOPPED = "stopped"
    ERROR = "error"


class CarbonDataSource(str, Enum):
    """Available carbon intensity data sources."""
    ELECTRICITYMAP = "electricitymap"
    WATTTIME = "watttime"
    CUSTOM = "custom"
    CACHED = "cached"


@dataclass
class CarbonIntensity:
    """Carbon intensity measurement at a specific time and location."""
    region: str
    timestamp: datetime
    carbon_intensity: float
    unit: CarbonIntensityUnit = CarbonIntensityUnit.GRAMS_CO2_PER_KWH
    data_source: Optional[str] = None
    confidence: Optional[float] = None
    renewable_percentage: Optional[float] = None


@dataclass
class EnergyMix:
    """Energy generation mix for a region at a specific time."""
    timestamp: datetime
    region: str
    solar: float = 0.0
    wind: float = 0.0
    hydro: float = 0.0
    nuclear: float = 0.0
    gas: float = 0.0
    coal: float = 0.0
    oil: float = 0.0
    biomass: float = 0.0
    geothermal: float = 0.0
    other: float = 0.0


@dataclass
class CarbonForecast:
    """Carbon intensity forecast for a region."""
    region: str
    forecast_start: datetime
    forecast_end: datetime
    data_points: List[CarbonIntensity]
    confidence_interval: Optional[float] = None
    model_name: Optional[str] = None


@dataclass
class TrainingMetrics:
    """Metrics for a carbon-aware training session."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_energy_kwh: float = 0.0
    total_carbon_kg: float = 0.0
    avg_carbon_intensity: float = 0.0
    peak_carbon_intensity: float = 0.0
    min_carbon_intensity: float = float('inf')
    paused_duration: timedelta = field(default_factory=lambda: timedelta(0))
    migrations: int = 0
    training_efficiency: Optional[float] = None
    carbon_saved_kg: float = 0.0


@dataclass
class OptimalWindow:
    """Optimal training window based on carbon forecast."""
    start_time: datetime
    end_time: datetime
    avg_carbon_intensity: float
    total_expected_carbon_kg: float
    confidence_score: float
    renewable_percentage: float
    region: str


@dataclass
class TrainingConfig:
    """Configuration for carbon-aware training."""
    carbon_threshold: float = 100.0  # Max carbon intensity (gCO2/kWh)
    pause_threshold: float = 150.0   # Carbon intensity to pause training
    resume_threshold: float = 80.0   # Carbon intensity to resume training
    check_interval: int = 300        # Check interval in seconds
    max_pause_duration: timedelta = field(default_factory=lambda: timedelta(hours=6))  # Max pause time
    migration_enabled: bool = False  # Enable cross-region migration
    preferred_regions: List[str] = field(default_factory=list)
    excluded_regions: List[str] = field(default_factory=list)
    carbon_data_source: CarbonDataSource = CarbonDataSource.ELECTRICITYMAP
    forecast_horizon: timedelta = field(default_factory=lambda: timedelta(hours=24))  # Forecast window


@dataclass  
class RegionInfo:
    """Information about a compute region."""
    region_code: str
    display_name: str
    country: str
    timezone: str
    renewable_percentage: float
    avg_carbon_intensity: float
    available_gpus: int = 0
    cost_per_gpu_hour: float = 0.0
    network_latency_ms: float = 0.0


@dataclass
class RegionConfig:
    """Configuration for a specific region in multi-region optimization."""
    region_code: str
    gpus: int
    cost_per_hour: float
    carbon_intensity: float = 0.0
    renewable_percentage: float = 0.0
    migration_bandwidth_gbps: float = 1.0
    enabled: bool = True