"""Carbon intensity monitoring and tracking."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
import logging
from ..core.types import (
    CarbonIntensity, CarbonForecast, OptimalWindow, 
    TrainingMetrics, CarbonDataSource
)
from ..carbon_models.base import CarbonDataProvider
from ..carbon_models.electricitymap import ElectricityMapProvider
from ..carbon_models.watttime import WattTimeProvider  
from ..carbon_models.cached import CachedProvider


logger = logging.getLogger(__name__)


class CarbonMonitor:
    """Real-time carbon intensity monitoring and optimization."""
    
    def __init__(
        self,
        regions: List[str],
        data_source: CarbonDataSource = CarbonDataSource.ELECTRICITYMAP,
        api_key: Optional[str] = None,
        update_interval: int = 300,
        cache_duration: int = 300
    ):
        """Initialize carbon monitor.
        
        Args:
            regions: List of regions to monitor
            data_source: Carbon data source to use
            api_key: API key for data source
            update_interval: Update interval in seconds
            cache_duration: Cache duration in seconds
        """
        self.regions = regions
        self.data_source = data_source
        self.update_interval = update_interval
        
        # Initialize data provider
        if data_source == CarbonDataSource.ELECTRICITYMAP:
            self.provider = ElectricityMapProvider(api_key, cache_duration)
        elif data_source == CarbonDataSource.WATTTIME:
            self.provider = WattTimeProvider(api_key, cache_duration)
        elif data_source == CarbonDataSource.CACHED:
            if not api_key:
                raise ValueError("Cached data source requires data file path as api_key")
            self.provider = CachedProvider(api_key, cache_duration)
        else:
            raise ValueError(f"Unsupported data source: {data_source}")
        
        # Monitoring state
        self._current_intensities: Dict[str, CarbonIntensity] = {}
        self._forecasts: Dict[str, CarbonForecast] = {}
        self._callbacks: List[Callable] = []
        self._monitoring_task: Optional[asyncio.Task] = None
        self._stop_monitoring = False
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.provider.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_monitoring()
        await self.provider.__aexit__(exc_type, exc_val, exc_tb)
    
    async def start_monitoring(self) -> None:
        """Start continuous carbon intensity monitoring."""
        if self._monitoring_task and not self._monitoring_task.done():
            logger.warning("Monitoring already started")
            return
        
        self._stop_monitoring = False
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"Started carbon monitoring for regions: {self.regions}")
    
    async def stop_monitoring(self) -> None:
        """Stop carbon intensity monitoring."""
        self._stop_monitoring = True
        
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped carbon monitoring")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self._stop_monitoring:
            try:
                # Update current intensities
                new_intensities = await self.provider.get_multiple_regions_intensity(
                    self.regions
                )
                
                # Check for significant changes
                for region, intensity in new_intensities.items():
                    old_intensity = self._current_intensities.get(region)
                    
                    if (not old_intensity or 
                        abs(intensity.carbon_intensity - old_intensity.carbon_intensity) > 10):
                        await self._notify_callbacks('intensity_change', {
                            'region': region,
                            'old_intensity': old_intensity,
                            'new_intensity': intensity
                        })
                
                self._current_intensities = new_intensities
                
                # Update forecasts periodically (every hour)
                current_time = datetime.now()
                should_update_forecasts = any(
                    not forecast or 
                    current_time - forecast.forecast_start > timedelta(hours=1)
                    for forecast in self._forecasts.values()
                )
                
                if should_update_forecasts:
                    await self._update_forecasts()
                
                logger.debug(f"Updated carbon intensities: {len(new_intensities)} regions")
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await self._notify_callbacks('monitoring_error', {'error': e})
            
            # Wait for next update
            await asyncio.sleep(self.update_interval)
    
    async def _update_forecasts(self) -> None:
        """Update carbon intensity forecasts."""
        for region in self.regions:
            try:
                forecast = await self.provider.get_forecast(
                    region, 
                    duration=timedelta(hours=48)
                )
                self._forecasts[region] = forecast
                
            except Exception as e:
                logger.error(f"Failed to update forecast for {region}: {e}")
    
    async def _notify_callbacks(self, event_type: str, data: Dict[str, Any]) -> None:
        """Notify registered callbacks of events."""
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_type, data)
                else:
                    callback(event_type, data)
            except Exception as e:
                logger.error(f"Error in callback: {e}")
    
    def add_callback(self, callback: Callable) -> None:
        """Add monitoring event callback.
        
        Args:
            callback: Callback function (event_type, data) -> None
        """
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable) -> None:
        """Remove monitoring event callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    async def get_current_intensity(self, region: str) -> Optional[CarbonIntensity]:
        """Get current carbon intensity for a region.
        
        Args:
            region: Region code
            
        Returns:
            Current carbon intensity or None if unavailable
        """
        if region in self._current_intensities:
            return self._current_intensities[region]
        
        try:
            intensity = await self.provider.get_current_intensity(region)
            self._current_intensities[region] = intensity
            return intensity
        except Exception as e:
            logger.error(f"Failed to get current intensity for {region}: {e}")
            return None
    
    async def get_forecast(
        self, 
        region: str, 
        hours: int = 24
    ) -> Optional[CarbonForecast]:
        """Get carbon intensity forecast for a region.
        
        Args:
            region: Region code
            hours: Forecast duration in hours
            
        Returns:
            Carbon intensity forecast or None if unavailable
        """
        try:
            forecast = await self.provider.get_forecast(
                region, 
                duration=timedelta(hours=hours)
            )
            self._forecasts[region] = forecast
            return forecast
        except Exception as e:
            logger.error(f"Failed to get forecast for {region}: {e}")
            return None
    
    def find_optimal_window(
        self,
        duration_hours: int,
        max_carbon_intensity: Optional[float] = None,
        preferred_regions: Optional[List[str]] = None
    ) -> Optional[OptimalWindow]:
        """Find optimal training window based on forecasts.
        
        Args:
            duration_hours: Required training duration in hours
            max_carbon_intensity: Maximum acceptable carbon intensity
            preferred_regions: Preferred regions (defaults to all monitored)
            
        Returns:
            Optimal training window or None if no suitable window found
        """
        if not preferred_regions:
            preferred_regions = self.regions
        
        best_window = None
        best_score = float('inf')
        
        for region in preferred_regions:
            forecast = self._forecasts.get(region)
            if not forecast or not forecast.data_points:
                continue
            
            # Find best contiguous window in forecast
            data_points = sorted(forecast.data_points, key=lambda x: x.timestamp)
            
            for i in range(len(data_points) - duration_hours + 1):
                window_points = data_points[i:i + duration_hours]
                
                # Check if window meets constraints
                avg_intensity = sum(p.carbon_intensity for p in window_points) / len(window_points)
                max_intensity = max(p.carbon_intensity for p in window_points)
                
                if max_carbon_intensity and max_intensity > max_carbon_intensity:
                    continue
                
                # Score window (lower is better)
                score = avg_intensity
                
                if score < best_score:
                    best_score = score
                    renewable_pct = sum(
                        p.renewable_percentage or 0 
                        for p in window_points
                    ) / len(window_points)
                    
                    best_window = OptimalWindow(
                        start_time=window_points[0].timestamp,
                        end_time=window_points[-1].timestamp,
                        avg_carbon_intensity=avg_intensity,
                        total_expected_carbon_kg=0.0,  # Will be calculated by trainer
                        confidence_score=1.0 - (score / 500),  # Normalized confidence
                        renewable_percentage=renewable_pct,
                        region=region
                    )
        
        return best_window
    
    def get_cleanest_region(self) -> Optional[str]:
        """Get region with lowest current carbon intensity.
        
        Returns:
            Region code with lowest carbon intensity
        """
        return self.provider.find_cleanest_region(self._current_intensities)
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current monitoring status.
        
        Returns:
            Dictionary with current status information
        """
        return {
            "regions": self.regions,
            "data_source": self.data_source.value,
            "monitoring_active": (
                self._monitoring_task and 
                not self._monitoring_task.done()
            ),
            "current_intensities": {
                region: {
                    "carbon_intensity": intensity.carbon_intensity,
                    "timestamp": intensity.timestamp.isoformat(),
                    "renewable_percentage": intensity.renewable_percentage
                }
                for region, intensity in self._current_intensities.items()
            },
            "forecasts_available": list(self._forecasts.keys()),
            "cleanest_region": self.get_cleanest_region()
        }