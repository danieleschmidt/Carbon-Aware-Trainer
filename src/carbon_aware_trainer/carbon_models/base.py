"""Base class for carbon intensity data providers."""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import asyncio
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    aiohttp = None
from ..core.types import CarbonIntensity, CarbonForecast, EnergyMix


class CarbonDataProvider(ABC):
    """Abstract base class for carbon intensity data sources."""
    
    def __init__(self, api_key: Optional[str] = None, cache_duration: int = 3600):
        """Initialize the carbon data provider.
        
        Args:
            api_key: API key for the service (if required)
            cache_duration: Cache duration in seconds
        """
        self.api_key = api_key
        self.cache_duration = cache_duration
        self._cache: Dict[str, Any] = {}
        self._session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        if HAS_AIOHTTP:
            self._session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()
    
    @abstractmethod
    async def get_current_intensity(self, region: str) -> CarbonIntensity:
        """Get current carbon intensity for a region.
        
        Args:
            region: Region code (e.g., 'US-CA', 'EU-FR')
            
        Returns:
            Current carbon intensity data
        """
        pass
    
    @abstractmethod
    async def get_forecast(
        self, 
        region: str, 
        start_time: Optional[datetime] = None,
        duration: timedelta = timedelta(hours=24)
    ) -> CarbonForecast:
        """Get carbon intensity forecast for a region.
        
        Args:
            region: Region code
            start_time: Forecast start time (defaults to now)
            duration: Forecast duration
            
        Returns:
            Carbon intensity forecast
        """
        pass
    
    @abstractmethod
    async def get_energy_mix(self, region: str) -> Optional[EnergyMix]:
        """Get current energy generation mix for a region.
        
        Args:
            region: Region code
            
        Returns:
            Energy mix data if available
        """
        pass
    
    @abstractmethod
    def get_supported_regions(self) -> List[str]:
        """Get list of supported region codes.
        
        Returns:
            List of supported region codes
        """
        pass
    
    def _cache_key(self, method: str, **kwargs) -> str:
        """Generate cache key for method and parameters."""
        sorted_kwargs = sorted(kwargs.items())
        key_parts = [method] + [f"{k}={v}" for k, v in sorted_kwargs]
        return "|".join(key_parts)
    
    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid."""
        if not cache_entry:
            return False
        
        cache_time = cache_entry.get('timestamp')
        if not cache_time:
            return False
            
        age = datetime.now() - cache_time
        return age.total_seconds() < self.cache_duration
    
    def _cache_get(self, cache_key: str) -> Optional[Any]:
        """Get value from cache if valid."""
        cache_entry = self._cache.get(cache_key)
        if self._is_cache_valid(cache_entry):
            return cache_entry['data']
        return None
    
    def _cache_set(self, cache_key: str, data: Any) -> None:
        """Set value in cache with timestamp."""
        self._cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    async def get_multiple_regions_intensity(
        self, regions: List[str]
    ) -> Dict[str, CarbonIntensity]:
        """Get current carbon intensity for multiple regions.
        
        Args:
            regions: List of region codes
            
        Returns:
            Dictionary mapping region codes to carbon intensity
        """
        tasks = [self.get_current_intensity(region) for region in regions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        intensity_map = {}
        for region, result in zip(regions, results):
            if not isinstance(result, Exception):
                intensity_map[region] = result
                
        return intensity_map
    
    def find_cleanest_region(
        self, intensities: Dict[str, CarbonIntensity]
    ) -> Optional[str]:
        """Find the region with lowest carbon intensity.
        
        Args:
            intensities: Map of region codes to carbon intensities
            
        Returns:
            Region code with lowest carbon intensity
        """
        if not intensities:
            return None
            
        return min(
            intensities.keys(),
            key=lambda region: intensities[region].carbon_intensity
        )