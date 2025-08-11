"""
Multi-region carbon-aware orchestration for distributed training.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
from pydantic import BaseModel

from .types import CarbonIntensity, CarbonForecast
from .monitor import CarbonMonitor
from .exceptions import CarbonAwareException


logger = logging.getLogger(__name__)


class RegionStatus(Enum):
    """Status of a training region."""
    AVAILABLE = "available"
    BUSY = "busy"
    MIGRATING = "migrating"
    OFFLINE = "offline"


@dataclass
class RegionConfig:
    """Configuration for a training region."""
    region_id: str
    gpus: int
    cost_per_hour: float
    bandwidth_gbps: float
    carbon_threshold: float = 100.0  # gCO2/kWh
    migration_cost_factor: float = 0.1  # Cost multiplier for migration


class PlacementPlan(BaseModel):
    """Optimal placement plan for multi-region training."""
    primary_region: str
    backup_regions: List[str]
    expected_carbon_kg: float
    expected_cost_usd: float
    estimated_duration_hours: float
    migration_windows: List[Dict[str, Any]]
    carbon_savings_pct: float


class MultiRegionOrchestrator:
    """Orchestrates carbon-aware training across multiple regions."""
    
    def __init__(
        self,
        regions: Dict[str, RegionConfig],
        monitor: Optional[CarbonMonitor] = None,
        migration_bandwidth_gbps: float = 10.0,
        checkpoint_size_gb: float = 5.0
    ):
        self.regions = {rid: RegionConfig(**config) if isinstance(config, dict) else config 
                       for rid, config in regions.items()}
        self.monitor = monitor or CarbonMonitor()
        self.migration_bandwidth_gbps = migration_bandwidth_gbps
        self.checkpoint_size_gb = checkpoint_size_gb
        self.region_status = {rid: RegionStatus.AVAILABLE for rid in regions.keys()}
        self._active_region = None
        self._migration_history = []
        
    async def optimize_placement(
        self,
        model_size_gb: float,
        dataset_size_gb: float,
        training_hours: float,
        carbon_budget_kg: Optional[float] = None,
        cost_budget_usd: Optional[float] = None
    ) -> PlacementPlan:
        """Optimize training placement across regions."""
        logger.info(f"Optimizing placement for {training_hours}h training")
        
        # Get carbon forecasts for all regions
        region_forecasts = {}
        for region_id in self.regions.keys():
            try:
                forecast = await self.monitor.get_forecast(region_id, hours=int(training_hours) + 24)
                region_forecasts[region_id] = forecast
            except Exception as e:
                logger.warning(f"Failed to get forecast for {region_id}: {e}")
                region_forecasts[region_id] = None
                
        # Calculate optimal placement
        best_plan = await self._calculate_optimal_plan(
            region_forecasts,
            model_size_gb,
            dataset_size_gb, 
            training_hours,
            carbon_budget_kg,
            cost_budget_usd
        )
        
        return best_plan
        
    async def _calculate_optimal_plan(
        self,
        region_forecasts: Dict[str, Optional[CarbonForecast]],
        model_size_gb: float,
        dataset_size_gb: float,
        training_hours: float,
        carbon_budget_kg: Optional[float],
        cost_budget_usd: Optional[float]
    ) -> PlacementPlan:
        """Calculate the optimal placement plan."""
        
        # Simple heuristic: choose region with lowest average carbon intensity
        region_scores = {}
        
        for region_id, config in self.regions.items():
            if region_forecasts.get(region_id) is None:
                continue
                
            forecast = region_forecasts[region_id]
            if not forecast or not forecast.data_points:
                # Use current intensity as fallback
                try:
                    current = await self.monitor.get_current_intensity(region_id)
                    avg_carbon = current.carbon_intensity if current else 200.0
                except:
                    avg_carbon = 200.0  # Conservative fallback
            else:
                avg_carbon = np.mean([dp.carbon_intensity for dp in forecast.data_points])
            
            # Calculate estimated costs and emissions
            gpu_hours = config.gpus * training_hours
            estimated_cost = gpu_hours * config.cost_per_hour
            estimated_carbon = (avg_carbon / 1000) * gpu_hours * 0.4  # Rough GPU power estimation
            
            # Simple scoring function (lower is better)
            carbon_score = avg_carbon / 100.0  # Normalize
            cost_score = estimated_cost / 1000.0  # Normalize
            region_scores[region_id] = carbon_score + cost_score * 0.3
            
        if not region_scores:
            raise CarbonAwareException("No regions available for placement")
            
        # Sort by score
        sorted_regions = sorted(region_scores.items(), key=lambda x: x[1])
        primary_region = sorted_regions[0][0]
        backup_regions = [r[0] for r in sorted_regions[1:3]]  # Top 2 backups
        
        # Calculate plan metrics
        primary_config = self.regions[primary_region]
        primary_forecast = region_forecasts[primary_region]
        
        if primary_forecast and primary_forecast.data_points:
            avg_carbon = np.mean([dp.carbon_intensity for dp in primary_forecast.data_points])
        else:
            avg_carbon = 150.0  # Default estimate
            
        gpu_hours = primary_config.gpus * training_hours
        expected_cost = gpu_hours * primary_config.cost_per_hour
        expected_carbon = (avg_carbon / 1000) * gpu_hours * 0.4
        
        # Calculate potential migration windows
        migration_windows = await self._find_migration_windows(
            region_forecasts, training_hours
        )
        
        # Estimate carbon savings vs. single-region training
        baseline_carbon = (200.0 / 1000) * gpu_hours * 0.4  # High-carbon baseline
        carbon_savings_pct = max(0, (baseline_carbon - expected_carbon) / baseline_carbon * 100)
        
        return PlacementPlan(
            primary_region=primary_region,
            backup_regions=backup_regions,
            expected_carbon_kg=expected_carbon,
            expected_cost_usd=expected_cost,
            estimated_duration_hours=training_hours,
            migration_windows=migration_windows,
            carbon_savings_pct=carbon_savings_pct
        )
        
    async def _find_migration_windows(
        self,
        region_forecasts: Dict[str, Optional[CarbonForecast]],
        training_hours: float
    ) -> List[Dict[str, Any]]:
        """Find optimal windows for cross-region migration."""
        migration_windows = []
        
        # Simple approach: find high-carbon periods in primary region
        for region_id, forecast in region_forecasts.items():
            if not forecast or not forecast.data_points:
                continue
                
            high_carbon_periods = []
            for i, dp in enumerate(forecast.data_points):
                if dp.carbon_intensity > self.regions[region_id].carbon_threshold:
                    high_carbon_periods.append({
                        'start_time': dp.timestamp,
                        'carbon_intensity': dp.carbon_intensity,
                        'duration_hours': 1
                    })
                    
            if high_carbon_periods:
                migration_windows.extend(high_carbon_periods[:3])  # Limit to 3 windows
                
        return migration_windows
        
    async def execute_training(
        self,
        placement_plan: PlacementPlan,
        training_function: callable,
        checkpoint_interval: timedelta = timedelta(hours=1),
        migration_threshold: float = 100.0
    ) -> Dict[str, Any]:
        """Execute training with dynamic region switching."""
        logger.info(f"Starting training in region: {placement_plan.primary_region}")
        
        self._active_region = placement_plan.primary_region
        self.region_status[self._active_region] = RegionStatus.BUSY
        
        training_metrics = {
            'start_time': datetime.now(),
            'regions_used': [self._active_region],
            'migrations': 0,
            'total_carbon_kg': 0.0,
            'total_cost_usd': 0.0
        }
        
        try:
            # Run training with periodic carbon checks
            while True:
                # Check current carbon intensity
                current_intensity = await self.monitor.get_current_intensity(self._active_region)
                
                if current_intensity and current_intensity.carbon_intensity > migration_threshold:
                    logger.info(f"High carbon intensity ({current_intensity.carbon_intensity}), considering migration")
                    
                    # Find best alternative region
                    best_alternative = await self._find_best_migration_target(
                        placement_plan.backup_regions,
                        migration_threshold
                    )
                    
                    if best_alternative:
                        await self._migrate_training(best_alternative, training_function)
                        training_metrics['migrations'] += 1
                        training_metrics['regions_used'].append(best_alternative)
                        self._active_region = best_alternative
                
                # Simulate training step (in real implementation, this would be the actual training)
                await asyncio.sleep(1)
                break  # For demo purposes, exit after one iteration
                
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Clean up
            if self._active_region:
                self.region_status[self._active_region] = RegionStatus.AVAILABLE
                
            training_metrics['end_time'] = datetime.now()
            training_metrics['duration_hours'] = (
                training_metrics['end_time'] - training_metrics['start_time']
            ).total_seconds() / 3600
            
        return training_metrics
        
    async def _find_best_migration_target(
        self,
        backup_regions: List[str],
        threshold: float
    ) -> Optional[str]:
        """Find the best region for migration."""
        for region_id in backup_regions:
            if self.region_status[region_id] != RegionStatus.AVAILABLE:
                continue
                
            try:
                current = await self.monitor.get_current_intensity(region_id)
                if current and current.carbon_intensity < threshold:
                    return region_id
            except Exception as e:
                logger.warning(f"Failed to check {region_id}: {e}")
                continue
                
        return None
        
    async def _migrate_training(
        self,
        target_region: str,
        training_function: callable
    ) -> None:
        """Migrate training to a different region."""
        if not self._active_region:
            raise CarbonAwareException("No active training to migrate")
            
        logger.info(f"Migrating training from {self._active_region} to {target_region}")
        
        # Mark regions appropriately
        self.region_status[self._active_region] = RegionStatus.AVAILABLE
        self.region_status[target_region] = RegionStatus.MIGRATING
        
        try:
            # Simulate checkpoint transfer (in real implementation, this would transfer actual checkpoints)
            migration_time = self.checkpoint_size_gb / self.migration_bandwidth_gbps * 3600  # seconds
            await asyncio.sleep(min(migration_time, 5))  # Cap at 5 seconds for demo
            
            # Record migration
            self._migration_history.append({
                'timestamp': datetime.now(),
                'from_region': self._active_region,
                'to_region': target_region,
                'reason': 'high_carbon_intensity'
            })
            
            logger.info(f"Migration completed to {target_region}")
            
        finally:
            self.region_status[target_region] = RegionStatus.BUSY
            
    def get_migration_history(self) -> List[Dict[str, Any]]:
        """Get history of region migrations."""
        return self._migration_history.copy()
        
    def get_region_status(self) -> Dict[str, RegionStatus]:
        """Get current status of all regions."""
        return self.region_status.copy()