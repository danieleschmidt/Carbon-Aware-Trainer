"""Comprehensive backup and fallback system for carbon data sources."""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import aiofiles
# import shutil

from .exceptions import CarbonDataError, CarbonProviderError
from .types import CarbonIntensity, CarbonForecast, CarbonDataSource
from ..carbon_models.base import CarbonDataProvider


logger = logging.getLogger(__name__)


class FallbackStrategy(str, Enum):
    """Fallback strategies for handling data source failures."""
    STATIC_VALUES = "static_values"
    CACHED_DATA = "cached_data"
    ALTERNATIVE_PROVIDER = "alternative_provider"
    INTERPOLATED_VALUES = "interpolated_values"
    REGIONAL_AVERAGE = "regional_average"


@dataclass
class FallbackConfig:
    """Configuration for fallback behavior."""
    enabled: bool = True
    strategies: List[FallbackStrategy] = field(default_factory=lambda: [
        FallbackStrategy.CACHED_DATA,
        FallbackStrategy.ALTERNATIVE_PROVIDER,
        FallbackStrategy.INTERPOLATED_VALUES,
        FallbackStrategy.STATIC_VALUES
    ])
    cache_expiry_hours: int = 24
    static_carbon_intensity: float = 400.0  # Conservative fallback value
    static_renewable_percentage: float = 20.0
    max_interpolation_gap_hours: int = 6
    enable_cross_region_fallback: bool = True


@dataclass
class BackupEntry:
    """Represents a backup data entry."""
    timestamp: datetime
    region: str
    carbon_intensity: float
    renewable_percentage: Optional[float] = None
    data_source: str = "backup"
    confidence: float = 0.5  # Lower confidence for backup data


class DataBackupManager:
    """Manages backup of carbon intensity data."""
    
    def __init__(self, backup_dir: Union[str, Path], retention_days: int = 30):
        """Initialize backup manager.
        
        Args:
            backup_dir: Directory to store backup files
            retention_days: How long to retain backup files
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.retention_days = retention_days
        self.retention_cutoff = timedelta(days=retention_days)
        
        # In-memory cache for recent data
        self.memory_cache: Dict[str, List[BackupEntry]] = {}
        self.max_memory_entries = 1000
        
        logger.info(f"DataBackupManager initialized with backup_dir: {backup_dir}")
    
    async def backup_carbon_data(
        self,
        region: str,
        carbon_intensity: float,
        renewable_percentage: Optional[float] = None,
        data_source: str = "unknown"
    ) -> None:
        """Backup carbon intensity data.
        
        Args:
            region: Region code
            carbon_intensity: Carbon intensity value
            renewable_percentage: Renewable energy percentage
            data_source: Source of the data
        """
        try:
            backup_entry = BackupEntry(
                timestamp=datetime.now(),
                region=region,
                carbon_intensity=carbon_intensity,
                renewable_percentage=renewable_percentage,
                data_source=data_source
            )
            
            # Add to memory cache
            if region not in self.memory_cache:
                self.memory_cache[region] = []
            
            self.memory_cache[region].append(backup_entry)
            
            # Trim memory cache if needed
            if len(self.memory_cache[region]) > self.max_memory_entries:
                self.memory_cache[region] = self.memory_cache[region][-self.max_memory_entries:]
            
            # Write to disk
            await self._write_backup_to_disk(backup_entry)
            
            logger.debug(f"Backed up carbon data for {region}: {carbon_intensity} gCO2/kWh")
            
        except Exception as e:
            logger.error(f"Failed to backup carbon data for {region}: {e}")
    
    async def get_backup_data(
        self,
        region: str,
        max_age_hours: int = 24,
        limit: int = 100
    ) -> List[BackupEntry]:
        """Get backup data for a region.
        
        Args:
            region: Region code
            max_age_hours: Maximum age of data to return
            limit: Maximum number of entries to return
            
        Returns:
            List of backup entries
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            # Get from memory cache first
            memory_data = []
            if region in self.memory_cache:
                memory_data = [
                    entry for entry in self.memory_cache[region]
                    if entry.timestamp >= cutoff_time
                ]
            
            # If we need more data, read from disk
            if len(memory_data) < limit:
                disk_data = await self._read_backup_from_disk(region, cutoff_time, limit)
                
                # Combine and deduplicate
                all_data = memory_data + disk_data
                seen_timestamps = set()
                unique_data = []
                
                for entry in sorted(all_data, key=lambda x: x.timestamp, reverse=True):
                    if entry.timestamp not in seen_timestamps:
                        unique_data.append(entry)
                        seen_timestamps.add(entry.timestamp)
                
                return unique_data[:limit]
            
            return sorted(memory_data, key=lambda x: x.timestamp, reverse=True)[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get backup data for {region}: {e}")
            return []
    
    async def get_latest_backup(self, region: str, max_age_hours: int = 24) -> Optional[BackupEntry]:
        """Get latest backup entry for a region.
        
        Args:
            region: Region code
            max_age_hours: Maximum age of data to consider
            
        Returns:
            Latest backup entry or None if not found
        """
        backup_data = await self.get_backup_data(region, max_age_hours, limit=1)
        return backup_data[0] if backup_data else None
    
    async def cleanup_old_backups(self) -> None:
        """Clean up old backup files."""
        try:
            cutoff_time = datetime.now() - self.retention_cutoff
            
            # Clean up disk files
            cleaned_files = 0
            for backup_file in self.backup_dir.glob("*.json"):
                try:
                    # Parse timestamp from filename
                    date_str = backup_file.stem.split('_')[0]  # Assumes format: YYYYMMDD_region.json
                    file_date = datetime.strptime(date_str, "%Y%m%d")
                    
                    if file_date < cutoff_time.replace(hour=0, minute=0, second=0, microsecond=0):
                        backup_file.unlink()
                        cleaned_files += 1
                        
                except (ValueError, IndexError):
                    # Skip files that don't match expected format
                    continue
            
            # Clean up memory cache
            for region in list(self.memory_cache.keys()):\n                entries = self.memory_cache[region]\n                self.memory_cache[region] = [\n                    entry for entry in entries if entry.timestamp >= cutoff_time\n                ]\n                \n                # Remove empty caches\n                if not self.memory_cache[region]:\n                    del self.memory_cache[region]
            
            if cleaned_files > 0:
                logger.info(f"Cleaned up {cleaned_files} old backup files")
                
        except Exception as e:
            logger.error(f"Error cleaning up backups: {e}")
    
    async def _write_backup_to_disk(self, backup_entry: BackupEntry) -> None:
        """Write backup entry to disk."""
        try:
            # Create daily backup file
            date_str = backup_entry.timestamp.strftime("%Y%m%d")
            filename = f"{date_str}_{backup_entry.region}.json"
            filepath = self.backup_dir / filename
            
            # Read existing data if file exists
            backup_data = []
            if filepath.exists():
                try:
                    async with aiofiles.open(filepath, 'r') as f:
                        content = await f.read()
                        backup_data = json.loads(content)
                except (json.JSONDecodeError, OSError):
                    logger.warning(f"Could not read existing backup file: {filepath}")
            
            # Add new entry
            backup_data.append({
                'timestamp': backup_entry.timestamp.isoformat(),
                'region': backup_entry.region,
                'carbon_intensity': backup_entry.carbon_intensity,
                'renewable_percentage': backup_entry.renewable_percentage,
                'data_source': backup_entry.data_source,
                'confidence': backup_entry.confidence
            })
            
            # Write updated data
            async with aiofiles.open(filepath, 'w') as f:
                await f.write(json.dumps(backup_data, indent=2))
                
        except Exception as e:
            logger.error(f"Failed to write backup to disk: {e}")
    
    async def _read_backup_from_disk(
        self,
        region: str,
        cutoff_time: datetime,
        limit: int
    ) -> List[BackupEntry]:
        """Read backup data from disk files."""
        try:
            backup_entries = []
            
            # Look through recent backup files
            for days_back in range(7):  # Look back up to 7 days
                date = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
                filename = f"{date}_{region}.json"
                filepath = self.backup_dir / filename
                
                if not filepath.exists():
                    continue
                
                try:
                    async with aiofiles.open(filepath, 'r') as f:
                        content = await f.read()
                        data = json.loads(content)
                    
                    for entry_data in data:
                        timestamp = datetime.fromisoformat(entry_data['timestamp'])
                        if timestamp >= cutoff_time:
                            entry = BackupEntry(
                                timestamp=timestamp,
                                region=entry_data['region'],
                                carbon_intensity=entry_data['carbon_intensity'],
                                renewable_percentage=entry_data.get('renewable_percentage'),
                                data_source=entry_data.get('data_source', 'backup'),
                                confidence=entry_data.get('confidence', 0.5)
                            )
                            backup_entries.append(entry)
                            
                except (json.JSONDecodeError, OSError, KeyError) as e:
                    logger.warning(f"Error reading backup file {filepath}: {e}")
                    continue
            
            return sorted(backup_entries, key=lambda x: x.timestamp, reverse=True)[:limit]
            
        except Exception as e:
            logger.error(f"Failed to read backup from disk: {e}")
            return []
    
    def get_backup_statistics(self) -> Dict[str, Any]:
        """Get backup system statistics.
        
        Returns:
            Dictionary with backup statistics
        """
        try:
            # Count backup files
            backup_files = list(self.backup_dir.glob("*.json"))
            
            # Count memory cache entries
            memory_entries = sum(len(entries) for entries in self.memory_cache.values())
            
            # Calculate disk usage
            disk_usage_mb = sum(f.stat().st_size for f in backup_files) / (1024 * 1024)
            
            return {
                'backup_files_count': len(backup_files),
                'memory_cache_regions': len(self.memory_cache),
                'memory_cache_entries': memory_entries,
                'disk_usage_mb': round(disk_usage_mb, 2),
                'backup_directory': str(self.backup_dir),
                'retention_days': self.retention_days
            }
            
        except Exception as e:
            logger.error(f"Error getting backup statistics: {e}")
            return {}


class FallbackDataProvider(CarbonDataProvider):
    """Fallback data provider that implements multiple fallback strategies."""
    
    def __init__(
        self,
        backup_manager: DataBackupManager,
        fallback_config: Optional[FallbackConfig] = None,
        primary_providers: Optional[List[CarbonDataProvider]] = None
    ):
        """Initialize fallback provider.
        
        Args:
            backup_manager: Data backup manager
            fallback_config: Fallback configuration
            primary_providers: List of primary providers to try first
        """
        super().__init__()
        self.backup_manager = backup_manager
        self.config = fallback_config or FallbackConfig()
        self.primary_providers = primary_providers or []
        
        # Regional averages for fallback
        self.regional_averages = {
            'US-CA': {'carbon_intensity': 250.0, 'renewable_percentage': 45.0},
            'US-TX': {'carbon_intensity': 450.0, 'renewable_percentage': 25.0},
            'EU-FR': {'carbon_intensity': 60.0, 'renewable_percentage': 75.0},
            'EU-DE': {'carbon_intensity': 350.0, 'renewable_percentage': 40.0},
            'EU-NO': {'carbon_intensity': 20.0, 'renewable_percentage': 95.0},
            'CN-BJ': {'carbon_intensity': 600.0, 'renewable_percentage': 15.0},
            'AU-NSW': {'carbon_intensity': 780.0, 'renewable_percentage': 20.0},
        }
        
        logger.info("FallbackDataProvider initialized")
    
    async def get_current_intensity(self, region: str) -> CarbonIntensity:
        """Get current carbon intensity with fallback strategies.
        
        Args:
            region: Region code
            
        Returns:
            Carbon intensity data
            
        Raises:
            CarbonDataError: If all fallback strategies fail
        """
        # Try primary providers first
        for provider in self.primary_providers:
            try:
                result = await provider.get_current_intensity(region)
                # Backup successful result
                await self.backup_manager.backup_carbon_data(
                    region,
                    result.carbon_intensity,
                    result.renewable_percentage,
                    getattr(provider, '__class__.__name__', 'unknown')
                )
                return result
            except Exception as e:
                logger.warning(f"Primary provider {provider.__class__.__name__} failed: {e}")
                continue
        
        # Try fallback strategies
        for strategy in self.config.strategies:
            try:
                result = await self._apply_fallback_strategy(strategy, region)
                if result:
                    logger.info(f"Using fallback strategy {strategy.value} for {region}")
                    return result
            except Exception as e:
                logger.warning(f"Fallback strategy {strategy.value} failed: {e}")
                continue
        
        raise CarbonDataError(f"All fallback strategies failed for region {region}")
    
    async def get_forecast(
        self,
        region: str,
        start_time: Optional[datetime] = None,
        duration: timedelta = timedelta(hours=24)
    ) -> CarbonForecast:
        """Get carbon intensity forecast with fallback.
        
        Args:
            region: Region code
            start_time: Forecast start time
            duration: Forecast duration
            
        Returns:
            Carbon forecast data
        """
        # Try primary providers
        for provider in self.primary_providers:
            try:
                return await provider.get_forecast(region, start_time, duration)
            except Exception as e:
                logger.warning(f"Primary provider forecast failed: {e}")
                continue
        
        # Fallback: create forecast based on historical data and current intensity
        try:
            current_intensity = await self.get_current_intensity(region)
            return await self._create_fallback_forecast(region, current_intensity, start_time, duration)
        except Exception as e:
            logger.error(f"Failed to create fallback forecast: {e}")
            raise CarbonDataError(f"No forecast available for {region}")
    
    async def get_energy_mix(self, region: str) -> Optional[Dict[str, float]]:
        """Get energy mix with fallback to regional averages."""
        # Try primary providers
        for provider in self.primary_providers:
            try:
                result = await provider.get_energy_mix(region)
                if result:
                    return result
            except Exception:
                continue
        
        # Fallback to estimated energy mix
        renewable_pct = self.regional_averages.get(region, {}).get('renewable_percentage', 30.0)
        fossil_pct = 100.0 - renewable_pct
        
        return {
            'renewable': renewable_pct,
            'fossil': fossil_pct,
            'nuclear': min(20.0, fossil_pct * 0.3),
            'unknown': 0.0
        }
    
    def get_supported_regions(self) -> List[str]:
        """Get list of supported regions."""
        # Combine regions from primary providers and fallback regions
        regions = set(self.regional_averages.keys())
        
        for provider in self.primary_providers:
            try:
                regions.update(provider.get_supported_regions())
            except Exception:
                continue
        
        return sorted(list(regions))
    
    async def _apply_fallback_strategy(
        self,
        strategy: FallbackStrategy,
        region: str
    ) -> Optional[CarbonIntensity]:
        """Apply specific fallback strategy.
        
        Args:
            strategy: Fallback strategy to apply
            region: Region code
            
        Returns:
            Carbon intensity data or None if strategy fails
        """
        if strategy == FallbackStrategy.CACHED_DATA:
            return await self._get_cached_data(region)
        
        elif strategy == FallbackStrategy.STATIC_VALUES:
            return self._get_static_values(region)
        
        elif strategy == FallbackStrategy.INTERPOLATED_VALUES:
            return await self._get_interpolated_values(region)
        
        elif strategy == FallbackStrategy.REGIONAL_AVERAGE:
            return self._get_regional_average(region)
        
        elif strategy == FallbackStrategy.ALTERNATIVE_PROVIDER:
            return await self._try_alternative_provider(region)
        
        return None
    
    async def _get_cached_data(self, region: str) -> Optional[CarbonIntensity]:
        """Get data from cache/backup."""
        latest_backup = await self.backup_manager.get_latest_backup(
            region,
            max_age_hours=self.config.cache_expiry_hours
        )
        
        if latest_backup:
            return CarbonIntensity(
                region=region,
                timestamp=latest_backup.timestamp,
                carbon_intensity=latest_backup.carbon_intensity,
                renewable_percentage=latest_backup.renewable_percentage,
                data_source="cached_backup",
                confidence=latest_backup.confidence
            )
        
        return None
    
    def _get_static_values(self, region: str) -> CarbonIntensity:
        """Get static fallback values."""
        return CarbonIntensity(
            region=region,
            timestamp=datetime.now(),
            carbon_intensity=self.config.static_carbon_intensity,
            renewable_percentage=self.config.static_renewable_percentage,
            data_source="static_fallback",
            confidence=0.1  # Very low confidence for static values
        )
    
    async def _get_interpolated_values(self, region: str) -> Optional[CarbonIntensity]:
        """Get interpolated values from historical data."""
        try:
            # Get recent backup data for interpolation
            backup_data = await self.backup_manager.get_backup_data(
                region,
                max_age_hours=self.config.max_interpolation_gap_hours * 2,
                limit=10
            )
            
            if len(backup_data) < 2:
                return None
            
            # Simple linear interpolation
            now = datetime.now()
            recent_entries = [
                entry for entry in backup_data
                if (now - entry.timestamp).total_seconds() / 3600 <= self.config.max_interpolation_gap_hours
            ]
            
            if not recent_entries:
                return None
            
            # Calculate weighted average based on time distance
            total_weight = 0
            weighted_intensity = 0
            
            for entry in recent_entries:
                hours_ago = (now - entry.timestamp).total_seconds() / 3600
                weight = 1.0 / (1.0 + hours_ago)  # More recent = higher weight
                
                weighted_intensity += entry.carbon_intensity * weight
                total_weight += weight
            
            if total_weight == 0:
                return None
            
            interpolated_intensity = weighted_intensity / total_weight
            
            return CarbonIntensity(
                region=region,
                timestamp=now,
                carbon_intensity=interpolated_intensity,
                renewable_percentage=recent_entries[0].renewable_percentage,
                data_source="interpolated",
                confidence=0.6
            )
            
        except Exception as e:
            logger.error(f"Interpolation failed for {region}: {e}")
            return None
    
    def _get_regional_average(self, region: str) -> Optional[CarbonIntensity]:
        """Get regional average values."""
        if region in self.regional_averages:
            avg_data = self.regional_averages[region]
            return CarbonIntensity(
                region=region,
                timestamp=datetime.now(),
                carbon_intensity=avg_data['carbon_intensity'],
                renewable_percentage=avg_data['renewable_percentage'],
                data_source="regional_average",
                confidence=0.4
            )
        
        return None
    
    async def _try_alternative_provider(self, region: str) -> Optional[CarbonIntensity]:
        """Try alternative data sources or providers."""
        # This could implement additional providers or data sources
        # For now, return None to continue with other strategies
        return None
    
    async def _create_fallback_forecast(
        self,
        region: str,
        current_intensity: CarbonIntensity,
        start_time: Optional[datetime],
        duration: timedelta
    ) -> CarbonForecast:
        """Create a fallback forecast based on current data and historical patterns."""
        start_time = start_time or datetime.now()
        end_time = start_time + duration
        
        # Get historical data for pattern analysis
        historical_data = await self.backup_manager.get_backup_data(
            region,
            max_age_hours=168,  # 7 days
            limit=168  # Hourly data for a week
        )
        
        # Create simple forecast points
        forecast_points = []
        hours = int(duration.total_seconds() / 3600)
        
        for hour in range(hours):
            forecast_time = start_time + timedelta(hours=hour)
            
            # Simple model: use current intensity with small random variations
            # In production, this would use more sophisticated forecasting
            base_intensity = current_intensity.carbon_intensity
            
            # Add daily pattern (lower at night, higher during day)
            hour_of_day = forecast_time.hour
            daily_factor = 0.8 + 0.4 * abs(12 - hour_of_day) / 12  # Peak around noon/midnight
            
            forecasted_intensity = base_intensity * daily_factor
            
            forecast_points.append(CarbonIntensity(
                region=region,
                timestamp=forecast_time,
                carbon_intensity=forecasted_intensity,
                renewable_percentage=current_intensity.renewable_percentage,
                data_source="fallback_forecast",
                confidence=0.3  # Low confidence for fallback forecast
            ))
        
        return CarbonForecast(
            region=region,
            forecast_start=start_time,
            forecast_end=end_time,
            data_points=forecast_points,
            confidence_interval=0.3,
            model_name="fallback_model"
        )


# Global backup manager and fallback provider
_backup_manager: Optional[DataBackupManager] = None
_fallback_provider: Optional[FallbackDataProvider] = None


def get_backup_manager(backup_dir: Optional[str] = None) -> DataBackupManager:
    """Get global backup manager instance."""
    global _backup_manager
    
    if _backup_manager is None:
        backup_dir = backup_dir or os.getenv('CARBON_AWARE_BACKUP_DIR', './backups')
        _backup_manager = DataBackupManager(backup_dir)
    
    return _backup_manager


def get_fallback_provider(
    primary_providers: Optional[List[CarbonDataProvider]] = None
) -> FallbackDataProvider:
    """Get global fallback provider instance."""
    global _fallback_provider
    
    if _fallback_provider is None:
        backup_manager = get_backup_manager()
        _fallback_provider = FallbackDataProvider(
            backup_manager=backup_manager,
            primary_providers=primary_providers or []
        )
    
    return _fallback_provider