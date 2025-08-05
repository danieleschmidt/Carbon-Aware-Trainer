"""Core type definitions for carbon-aware training."""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from pydantic import BaseModel, Field


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


class CarbonForecast(BaseModel):
    """Carbon intensity forecast for a region."""
    region: str
    forecast_start: datetime
    forecast_end: datetime
    data_points: List[CarbonIntensity]
    confidence_interval: Optional[float] = None
    model_name: Optional[str] = None


class TrainingMetrics(BaseModel):
    """Metrics for a carbon-aware training session."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_energy_kwh: float = 0.0
    total_carbon_kg: float = 0.0
    avg_carbon_intensity: float = 0.0
    peak_carbon_intensity: float = 0.0
    min_carbon_intensity: float = float('inf')
    paused_duration: timedelta = timedelta(0)
    migrations: int = 0
    training_efficiency: Optional[float] = None
    carbon_saved_kg: float = 0.0


class OptimalWindow(BaseModel):
    """Optimal training window based on carbon forecast."""
    start_time: datetime
    end_time: datetime
    avg_carbon_intensity: float
    total_expected_carbon_kg: float
    confidence_score: float
    renewable_percentage: float
    region: str


class TrainingConfig(BaseModel):
    """Configuration for carbon-aware training."""
    carbon_threshold: float = Field(100.0, description="Max carbon intensity (gCO2/kWh)")
    pause_threshold: float = Field(150.0, description="Carbon intensity to pause training")
    resume_threshold: float = Field(80.0, description="Carbon intensity to resume training")
    check_interval: int = Field(300, description="Check interval in seconds")
    max_pause_duration: timedelta = Field(timedelta(hours=6), description="Max pause time")
    migration_enabled: bool = Field(False, description="Enable cross-region migration")
    preferred_regions: List[str] = Field(default_factory=list)
    excluded_regions: List[str] = Field(default_factory=list)
    carbon_data_source: CarbonDataSource = CarbonDataSource.ELECTRICITYMAP
    forecast_horizon: timedelta = Field(timedelta(hours=24), description="Forecast window")


class RegionInfo(BaseModel):
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