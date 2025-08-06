"""Cached/offline carbon intensity provider."""

import json
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Union
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None
from .base import CarbonDataProvider
from ..core.types import CarbonIntensity, CarbonForecast, EnergyMix, CarbonIntensityUnit


class CachedProvider(CarbonDataProvider):
    """Provider that uses cached/offline carbon intensity data."""
    
    def __init__(self, data_source: Union[str, Path, Dict], cache_duration: int = 0):
        """Initialize cached provider.
        
        Args:
            data_source: Path to data file or dictionary with cached data
            cache_duration: Cache duration (unused for offline data)
        """
        super().__init__(None, cache_duration)
        
        if isinstance(data_source, (str, Path)):
            self.data = self._load_data_file(Path(data_source))
        else:
            self.data = data_source
        
        self._validate_data()
    
    def _load_data_file(self, file_path: Path) -> Dict:
        """Load carbon data from file."""
        if not file_path.exists():
            raise FileNotFoundError(f"Carbon data file not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix == ".json":
            with open(file_path, 'r') as f:
                return json.load(f)
        
        elif suffix == ".csv":
            return self._load_csv_data(file_path)
        
        elif suffix in [".parquet", ".pq"]:
            return self._load_parquet_data(file_path)
        
        else:
            raise ValueError(f"Unsupported data file format: {suffix}")
    
    def _load_csv_data(self, file_path: Path) -> Dict:
        """Load carbon data from CSV file."""
        if not HAS_PANDAS:
            raise ImportError("pandas required for CSV loading. pip install pandas")
        df = pd.read_csv(file_path)
        
        # Expected columns: region, timestamp, carbon_intensity
        required_cols = ["region", "timestamp", "carbon_intensity"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        
        data = {"regions": {}}
        
        for _, row in df.iterrows():
            region = row["region"]
            if region not in data["regions"]:
                data["regions"][region] = {"historical": [], "forecast": []}
            
            data["regions"][region]["historical"].append({
                "timestamp": row["timestamp"],
                "carbon_intensity": row["carbon_intensity"],
                "renewable_percentage": row.get("renewable_percentage"),
                "confidence": row.get("confidence")
            })
        
        return data
    
    def _load_parquet_data(self, file_path: Path) -> Dict:
        """Load carbon data from Parquet file."""
        if not HAS_PANDAS:
            raise ImportError("pandas required for Parquet loading. pip install pandas")
        df = pd.read_parquet(file_path)
        
        required_cols = ["region", "timestamp", "carbon_intensity"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Parquet must contain columns: {required_cols}")
        
        data = {"regions": {}}
        
        for _, row in df.iterrows():
            region = row["region"]
            if region not in data["regions"]:
                data["regions"][region] = {"historical": [], "forecast": []}
            
            data["regions"][region]["historical"].append({
                "timestamp": row["timestamp"].isoformat() if hasattr(row["timestamp"], "isoformat") else str(row["timestamp"]),
                "carbon_intensity": float(row["carbon_intensity"]),
                "renewable_percentage": float(row.get("renewable_percentage", 0)) if pd.notna(row.get("renewable_percentage")) else None,
                "confidence": float(row.get("confidence", 0)) if pd.notna(row.get("confidence")) else None
            })
        
        return data
    
    def _validate_data(self) -> None:
        """Validate loaded data structure."""
        if "regions" not in self.data:
            raise ValueError("Data must contain 'regions' key")
        
        if not isinstance(self.data["regions"], dict):
            raise ValueError("'regions' must be a dictionary")
    
    async def get_current_intensity(self, region: str) -> CarbonIntensity:
        """Get current carbon intensity from cached data.
        
        Uses the most recent data point available for the region.
        """
        if region not in self.data["regions"]:
            raise ValueError(f"Region {region} not found in cached data")
        
        region_data = self.data["regions"][region]
        historical = region_data.get("historical", [])
        
        if not historical:
            raise ValueError(f"No historical data available for region {region}")
        
        # Sort by timestamp and get most recent
        sorted_data = sorted(historical, key=lambda x: x["timestamp"], reverse=True)
        latest = sorted_data[0]
        
        return CarbonIntensity(
            region=region,
            timestamp=datetime.fromisoformat(latest["timestamp"].replace("Z", "+00:00")),
            carbon_intensity=latest["carbon_intensity"],
            unit=CarbonIntensityUnit.GRAMS_CO2_PER_KWH,
            data_source="cached",
            confidence=latest.get("confidence"),
            renewable_percentage=latest.get("renewable_percentage")
        )
    
    async def get_forecast(
        self, 
        region: str, 
        start_time: Optional[datetime] = None,
        duration: timedelta = timedelta(hours=24)
    ) -> CarbonForecast:
        """Get carbon intensity forecast from cached data."""
        if region not in self.data["regions"]:
            raise ValueError(f"Region {region} not found in cached data")
        
        if not start_time:
            start_time = datetime.now()
        
        end_time = start_time + duration
        
        region_data = self.data["regions"][region]
        forecast_data = region_data.get("forecast", [])
        
        # If no forecast data, use historical data as proxy
        if not forecast_data:
            forecast_data = region_data.get("historical", [])
        
        # Filter data within time range
        data_points = []
        for item in forecast_data:
            item_time = datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00"))
            if start_time <= item_time <= end_time:
                data_points.append(CarbonIntensity(
                    region=region,
                    timestamp=item_time,
                    carbon_intensity=item["carbon_intensity"],
                    unit=CarbonIntensityUnit.GRAMS_CO2_PER_KWH,
                    data_source="cached",
                    confidence=item.get("confidence"),
                    renewable_percentage=item.get("renewable_percentage")
                ))
        
        return CarbonForecast(
            region=region,
            forecast_start=start_time,
            forecast_end=end_time,
            data_points=data_points,
            model_name="cached"
        )
    
    async def get_energy_mix(self, region: str) -> Optional[EnergyMix]:
        """Get energy mix from cached data if available."""
        if region not in self.data["regions"]:
            return None
        
        region_data = self.data["regions"][region]
        energy_mix_data = region_data.get("energy_mix")
        
        if not energy_mix_data:
            return None
        
        # Get most recent energy mix data
        if isinstance(energy_mix_data, list):
            energy_mix_data = max(energy_mix_data, key=lambda x: x["timestamp"])
        
        return EnergyMix(
            timestamp=datetime.fromisoformat(energy_mix_data["timestamp"].replace("Z", "+00:00")),
            region=region,
            solar=energy_mix_data.get("solar", 0),
            wind=energy_mix_data.get("wind", 0),
            hydro=energy_mix_data.get("hydro", 0),
            nuclear=energy_mix_data.get("nuclear", 0),
            gas=energy_mix_data.get("gas", 0),
            coal=energy_mix_data.get("coal", 0),
            oil=energy_mix_data.get("oil", 0),
            biomass=energy_mix_data.get("biomass", 0),
            geothermal=energy_mix_data.get("geothermal", 0),
            other=energy_mix_data.get("other", 0)
        )
    
    def get_supported_regions(self) -> List[str]:
        """Get list of regions available in cached data."""
        return list(self.data["regions"].keys())
    
    @classmethod
    def create_sample_data(cls, output_path: Path, regions: List[str]) -> None:
        """Create sample carbon intensity data file for testing.
        
        Args:
            output_path: Path to save sample data
            regions: List of regions to include
        """
        import random
        from datetime import datetime, timedelta
        
        sample_data = {"regions": {}}
        
        for region in regions:
            historical = []
            forecast = []
            
            # Generate 7 days of historical data
            start_time = datetime.now() - timedelta(days=7)
            for i in range(7 * 24):  # Hourly data
                timestamp = start_time + timedelta(hours=i)
                
                # Simulate realistic carbon intensity patterns
                base_intensity = random.uniform(50, 200)
                daily_variation = 30 * abs(0.5 - (timestamp.hour / 24))
                seasonal_variation = 20 * random.random()
                
                intensity = base_intensity + daily_variation + seasonal_variation
                
                historical.append({
                    "timestamp": timestamp.isoformat(),
                    "carbon_intensity": round(intensity, 2),
                    "renewable_percentage": random.uniform(20, 80),
                    "confidence": random.uniform(0.7, 1.0)
                })
            
            # Generate 24 hours of forecast data
            forecast_start = datetime.now()
            for i in range(24):
                timestamp = forecast_start + timedelta(hours=i)
                intensity = random.uniform(40, 180)
                
                forecast.append({
                    "timestamp": timestamp.isoformat(),
                    "carbon_intensity": round(intensity, 2),
                    "renewable_percentage": random.uniform(25, 75),
                    "confidence": random.uniform(0.6, 0.9)
                })
            
            sample_data["regions"][region] = {
                "historical": historical,
                "forecast": forecast
            }
        
        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix.lower() == ".json":
            with open(output_path, 'w') as f:
                json.dump(sample_data, f, indent=2)
        else:
            raise ValueError("Only JSON output supported for sample data")