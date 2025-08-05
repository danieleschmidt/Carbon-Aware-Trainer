"""ElectricityMap API provider for carbon intensity data."""

import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import aiohttp
from .base import CarbonDataProvider
from ..core.types import CarbonIntensity, CarbonForecast, EnergyMix, CarbonIntensityUnit


class ElectricityMapProvider(CarbonDataProvider):
    """ElectricityMap API provider for real-time carbon intensity data."""
    
    BASE_URL = "https://api.electricitymap.org/v3"
    
    def __init__(self, api_key: Optional[str] = None, cache_duration: int = 300):
        """Initialize ElectricityMap provider.
        
        Args:
            api_key: ElectricityMap API key (or from ELECTRICITYMAP_API_KEY env var)
            cache_duration: Cache duration in seconds (default: 5 minutes)
        """
        if not api_key:
            api_key = os.getenv("ELECTRICITYMAP_API_KEY")
            
        if not api_key:
            raise ValueError(
                "ElectricityMap API key required. Set ELECTRICITYMAP_API_KEY "
                "environment variable or pass api_key parameter."
            )
            
        super().__init__(api_key, cache_duration)
        self.headers = {"auth-token": self.api_key}
    
    async def get_current_intensity(self, region: str) -> CarbonIntensity:
        """Get current carbon intensity from ElectricityMap.
        
        Args:
            region: Region code (e.g., 'US-CA', 'FR', 'DE')
            
        Returns:
            Current carbon intensity data
        """
        cache_key = self._cache_key("current_intensity", region=region)
        cached_result = self._cache_get(cache_key)
        if cached_result:
            return cached_result
        
        if not self._session:
            raise RuntimeError("Provider must be used as async context manager")
        
        url = f"{self.BASE_URL}/carbon-intensity/latest"
        params = {"zone": region}
        
        async with self._session.get(url, headers=self.headers, params=params) as response:
            response.raise_for_status()
            data = await response.json()
        
        intensity = CarbonIntensity(
            region=region,
            timestamp=datetime.fromisoformat(data["datetime"].replace("Z", "+00:00")),
            carbon_intensity=data["carbonIntensity"],
            unit=CarbonIntensityUnit.GRAMS_CO2_PER_KWH,
            data_source="electricitymap",
            renewable_percentage=self._calculate_renewable_percentage(
                data.get("powerConsumptionBreakdown", {})
            )
        )
        
        self._cache_set(cache_key, intensity)
        return intensity
    
    async def get_forecast(
        self, 
        region: str, 
        start_time: Optional[datetime] = None,
        duration: timedelta = timedelta(hours=24)
    ) -> CarbonForecast:
        """Get carbon intensity forecast from ElectricityMap.
        
        Args:
            region: Region code
            start_time: Forecast start time (defaults to now)
            duration: Forecast duration
            
        Returns:
            Carbon intensity forecast
        """
        if not start_time:
            start_time = datetime.now()
        
        end_time = start_time + duration
        
        cache_key = self._cache_key(
            "forecast", 
            region=region, 
            start=start_time.isoformat(),
            end=end_time.isoformat()
        )
        cached_result = self._cache_get(cache_key)
        if cached_result:
            return cached_result
        
        if not self._session:
            raise RuntimeError("Provider must be used as async context manager")
        
        url = f"{self.BASE_URL}/carbon-intensity/forecast"
        params = {
            "zone": region,
            "start": start_time.isoformat(),
            "end": end_time.isoformat()
        }
        
        async with self._session.get(url, headers=self.headers, params=params) as response:
            response.raise_for_status()
            data = await response.json()
        
        data_points = []
        for item in data.get("forecast", []):
            data_points.append(CarbonIntensity(
                region=region,
                timestamp=datetime.fromisoformat(item["datetime"].replace("Z", "+00:00")),
                carbon_intensity=item["carbonIntensity"],
                unit=CarbonIntensityUnit.GRAMS_CO2_PER_KWH,
                data_source="electricitymap"
            ))
        
        forecast = CarbonForecast(
            region=region,
            forecast_start=start_time,
            forecast_end=end_time,
            data_points=data_points,
            model_name="electricitymap"
        )
        
        self._cache_set(cache_key, forecast)
        return forecast
    
    async def get_energy_mix(self, region: str) -> Optional[EnergyMix]:
        """Get current energy generation mix from ElectricityMap.
        
        Args:
            region: Region code
            
        Returns:
            Energy mix data if available
        """
        cache_key = self._cache_key("energy_mix", region=region)
        cached_result = self._cache_get(cache_key)
        if cached_result:
            return cached_result
        
        if not self._session:
            raise RuntimeError("Provider must be used as async context manager")
        
        url = f"{self.BASE_URL}/power-breakdown/latest"
        params = {"zone": region}
        
        try:
            async with self._session.get(url, headers=self.headers, params=params) as response:
                response.raise_for_status()
                data = await response.json()
        except aiohttp.ClientError:
            return None
        
        breakdown = data.get("powerConsumptionBreakdown", {})
        
        energy_mix = EnergyMix(
            timestamp=datetime.fromisoformat(data["datetime"].replace("Z", "+00:00")),
            region=region,
            solar=breakdown.get("solar", 0) or 0,
            wind=breakdown.get("wind", 0) or 0,
            hydro=breakdown.get("hydro", 0) or 0,
            nuclear=breakdown.get("nuclear", 0) or 0,
            gas=breakdown.get("gas", 0) or 0,
            coal=breakdown.get("coal", 0) or 0,
            oil=breakdown.get("oil", 0) or 0,
            biomass=breakdown.get("biomass", 0) or 0,
            geothermal=breakdown.get("geothermal", 0) or 0,
            other=breakdown.get("unknown", 0) or 0
        )
        
        self._cache_set(cache_key, energy_mix)
        return energy_mix
    
    def get_supported_regions(self) -> List[str]:
        """Get list of supported ElectricityMap region codes."""
        return [
            # Major regions commonly used for ML training
            "US-CA", "US-NWPP", "US-SPP", "US-MISO", "US-PJM", "US-ISONE", "US-NYISO",
            "EU-FR", "EU-DE", "EU-GB", "EU-NL", "EU-IT", "EU-ES", "EU-SE", "EU-NO",
            "CN", "JP", "KR", "IN-KA", "IN-DL", "IN-MH", "IN-TN",
            "AU-NSW", "AU-VIC", "AU-QLD", "AU-SA", "AU-WA",
            "BR-S", "BR-SE", "BR-NE", "CA-AB", "CA-BC", "CA-ON", "CA-QC"
        ]
    
    def _calculate_renewable_percentage(self, breakdown: Dict) -> Optional[float]:
        """Calculate renewable energy percentage from power breakdown."""
        if not breakdown:
            return None
        
        renewable_sources = ["solar", "wind", "hydro", "biomass", "geothermal"]
        total_renewable = sum(
            breakdown.get(source, 0) or 0 
            for source in renewable_sources
        )
        
        total_consumption = sum(
            breakdown.get(source, 0) or 0 
            for source in breakdown.keys()
            if breakdown.get(source) is not None
        )
        
        if total_consumption == 0:
            return None
            
        return (total_renewable / total_consumption) * 100