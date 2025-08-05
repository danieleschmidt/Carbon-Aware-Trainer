"""WattTime API provider for carbon intensity data."""

import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import aiohttp
from .base import CarbonDataProvider
from ..core.types import CarbonIntensity, CarbonForecast, EnergyMix, CarbonIntensityUnit


class WattTimeProvider(CarbonDataProvider):
    """WattTime API provider for carbon intensity data."""
    
    BASE_URL = "https://api2.watttime.org/v2"
    
    def __init__(self, api_key: Optional[str] = None, cache_duration: int = 300):
        """Initialize WattTime provider.
        
        Args:
            api_key: WattTime API key (or from WATTTIME_API_KEY env var)
            cache_duration: Cache duration in seconds (default: 5 minutes)
        """
        if not api_key:
            api_key = os.getenv("WATTTIME_API_KEY")
            
        if not api_key:
            raise ValueError(
                "WattTime API key required. Set WATTTIME_API_KEY "
                "environment variable or pass api_key parameter."
            )
            
        super().__init__(api_key, cache_duration)
        self._access_token: Optional[str] = None
        self._token_expires: Optional[datetime] = None
    
    async def _get_access_token(self) -> str:
        """Get or refresh WattTime access token."""
        if (self._access_token and self._token_expires and 
            datetime.now() < self._token_expires - timedelta(minutes=5)):
            return self._access_token
        
        if not self._session:
            raise RuntimeError("Provider must be used as async context manager")
        
        # Parse username and password from API key
        if ":" not in self.api_key:
            raise ValueError("WattTime API key must be in format 'username:password'")
        
        username, password = self.api_key.split(":", 1)
        
        url = f"{self.BASE_URL}/login"
        auth = aiohttp.BasicAuth(username, password)
        
        async with self._session.get(url, auth=auth) as response:
            response.raise_for_status()
            data = await response.json()
        
        self._access_token = data["token"]
        self._token_expires = datetime.now() + timedelta(hours=1)
        
        return self._access_token
    
    async def get_current_intensity(self, region: str) -> CarbonIntensity:
        """Get current carbon intensity from WattTime.
        
        Args:
            region: Region code (balancing authority abbreviation)
            
        Returns:
            Current carbon intensity data
        """
        cache_key = self._cache_key("current_intensity", region=region)
        cached_result = self._cache_get(cache_key)
        if cached_result:
            return cached_result
        
        token = await self._get_access_token()
        headers = {"Authorization": f"Bearer {token}"}
        
        url = f"{self.BASE_URL}/index"
        params = {"ba": region}
        
        async with self._session.get(url, headers=headers, params=params) as response:
            response.raise_for_status()
            data = await response.json()
        
        intensity = CarbonIntensity(
            region=region,
            timestamp=datetime.fromisoformat(data["point_time"].replace("Z", "+00:00")),
            carbon_intensity=data["value"],
            unit=CarbonIntensityUnit.LBS_CO2_PER_MWH,
            data_source="watttime",
            confidence=data.get("frequency", None)
        )
        
        self._cache_set(cache_key, intensity)
        return intensity
    
    async def get_forecast(
        self, 
        region: str, 
        start_time: Optional[datetime] = None,
        duration: timedelta = timedelta(hours=24)
    ) -> CarbonForecast:
        """Get carbon intensity forecast from WattTime.
        
        Args:
            region: Region code (balancing authority abbreviation)
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
        
        token = await self._get_access_token()
        headers = {"Authorization": f"Bearer {token}"}
        
        url = f"{self.BASE_URL}/forecast"
        params = {
            "ba": region,
            "starttime": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "endtime": end_time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        
        async with self._session.get(url, headers=headers, params=params) as response:
            response.raise_for_status()
            data = await response.json()
        
        data_points = []
        for item in data.get("forecast", []):
            data_points.append(CarbonIntensity(
                region=region,
                timestamp=datetime.fromisoformat(item["point_time"].replace("Z", "+00:00")),
                carbon_intensity=item["value"],
                unit=CarbonIntensityUnit.LBS_CO2_PER_MWH,
                data_source="watttime",
                confidence=item.get("frequency", None)
            ))
        
        forecast = CarbonForecast(
            region=region,
            forecast_start=start_time,
            forecast_end=end_time,
            data_points=data_points,
            model_name="watttime"
        )
        
        self._cache_set(cache_key, forecast)
        return forecast
    
    async def get_energy_mix(self, region: str) -> Optional[EnergyMix]:
        """Get energy generation mix. WattTime doesn't provide this data."""
        return None
    
    def get_supported_regions(self) -> List[str]:
        """Get list of supported WattTime balancing authority codes."""
        return [
            # US Balancing Authorities commonly used for ML training
            "CAISO", "MISO", "PJM", "ERCOT", "SPP", "ISONE", "NYISO", "BPAT",
            "PACE", "PACW", "NEVP", "AZPS", "SRP", "WALC", "WACM", "CHPD",
            "DOPD", "GCPD", "GRID", "GRIF", "GRIS", "HGMA", "IPCO", "LDWP",
            "NWMT", "PACE", "PACW", "PGE", "PNM", "PSCO", "PSEI", "SCL",
            "SPPC", "TPWR", "WACM", "WALC", "WAUW"
        ]