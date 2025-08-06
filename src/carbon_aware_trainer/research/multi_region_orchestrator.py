"""Multi-region orchestration for carbon-optimized training."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from ..core.scheduler import CarbonAwareTrainer
from ..core.monitor import CarbonMonitor
from ..core.types import CarbonIntensity, OptimalWindow, TrainingState, CarbonDataSource

logger = logging.getLogger(__name__)


@dataclass
class RegionConfig:
    """Configuration for a training region."""
    
    region_code: str
    max_gpus: int
    cost_per_gpu_hour: float
    transfer_bandwidth_gbps: float = 10.0
    setup_time_minutes: int = 15
    availability_score: float = 1.0  # 0-1, higher is better


@dataclass
class MigrationPlan:
    """Plan for migrating training between regions."""
    
    from_region: str
    to_region: str
    start_time: datetime
    estimated_duration_minutes: int
    carbon_benefit_kg: float
    cost_impact_usd: float
    risk_score: float  # 0-1, lower is better


class MultiRegionOrchestrator:
    """Orchestrates carbon-aware training across multiple regions."""
    
    def __init__(
        self,
        regions: Dict[str, RegionConfig],
        carbon_budget_kg: Optional[float] = None,
        cost_budget_usd: Optional[float] = None,
        migration_threshold_gco2: float = 50.0,
        min_migration_benefit_kg: float = 5.0
    ):
        """Initialize multi-region orchestrator.
        
        Args:
            regions: Dictionary of region configurations
            carbon_budget_kg: Total carbon budget for training
            cost_budget_usd: Total cost budget for training
            migration_threshold_gco2: Trigger migration above this carbon intensity
            min_migration_benefit_kg: Minimum carbon savings to justify migration
        """
        self.regions = regions
        self.carbon_budget_kg = carbon_budget_kg
        self.cost_budget_usd = cost_budget_usd
        self.migration_threshold = migration_threshold_gco2
        self.min_migration_benefit_kg = min_migration_benefit_kg
        
        # Runtime state
        self.current_region: Optional[str] = None
        self.active_trainers: Dict[str, CarbonAwareTrainer] = {}
        self.monitors: Dict[str, CarbonMonitor] = {}
        self.migration_history: List[MigrationPlan] = []
        
        # Metrics tracking
        self.total_carbon_used_kg = 0.0
        self.total_cost_used_usd = 0.0
        self.total_migrations = 0
    
    async def initialize(self) -> None:
        """Initialize monitoring for all regions."""
        logger.info(f"Initializing multi-region orchestrator for {len(self.regions)} regions")
        
        for region_code in self.regions.keys():
            # Initialize carbon monitor for each region
            monitor = CarbonMonitor(
                regions=[region_code],
                data_source=CarbonDataSource.ELECTRICITYMAP,
                update_interval=300  # 5 minutes
            )
            
            await monitor.__aenter__()
            await monitor.start_monitoring()
            self.monitors[region_code] = monitor
        
        logger.info("Multi-region monitoring initialized")
    
    async def cleanup(self) -> None:
        """Cleanup all resources."""
        # Stop all monitors
        for monitor in self.monitors.values():
            await monitor.stop_monitoring()
            await monitor.__aexit__(None, None, None)
        
        # Cleanup all trainers
        for trainer in self.active_trainers.values():
            await trainer.cleanup()
        
        logger.info("Multi-region orchestrator cleaned up")
    
    async def find_optimal_region(
        self,
        training_duration_hours: int,
        required_gpus: int = 1,
        flexibility_hours: int = 48
    ) -> Tuple[str, OptimalWindow]:
        """Find the optimal region and time window for training.
        
        Args:
            training_duration_hours: Required training duration
            required_gpus: Number of GPUs needed
            flexibility_hours: Time flexibility for scheduling
            
        Returns:
            Tuple of (region_code, optimal_window)
        """
        best_region = None
        best_window = None
        best_score = float('inf')
        
        for region_code, region_config in self.regions.items():
            if region_config.max_gpus < required_gpus:
                continue
            
            monitor = self.monitors.get(region_code)
            if not monitor:
                continue
            
            # Find optimal window for this region
            window = monitor.find_optimal_window(
                duration_hours=training_duration_hours,
                max_carbon_intensity=self.migration_threshold,
                preferred_regions=[region_code]
            )
            
            if not window:
                continue
            
            # Calculate composite score (lower is better)
            carbon_score = window.avg_carbon_intensity
            cost_score = region_config.cost_per_gpu_hour * required_gpus * training_duration_hours
            availability_score = 1.0 / region_config.availability_score
            
            # Weighted composite score
            composite_score = (
                carbon_score * 0.5 +  # 50% weight on carbon
                cost_score * 0.3 +    # 30% weight on cost
                availability_score * 0.2  # 20% weight on availability
            )
            
            if composite_score < best_score:
                best_score = composite_score
                best_region = region_code
                best_window = window
        
        if not best_region:
            raise RuntimeError("No suitable region found for training requirements")
        
        logger.info(f"Optimal region selected: {best_region} (score: {best_score:.2f})")
        return best_region, best_window
    
    async def start_training_in_region(
        self,
        region: str,
        model: Any,
        optimizer: Any,
        target_carbon_intensity: float = 100.0
    ) -> CarbonAwareTrainer:
        """Start carbon-aware training in specified region.
        
        Args:
            region: Region to start training in
            model: ML model to train
            optimizer: Model optimizer
            target_carbon_intensity: Carbon intensity threshold
            
        Returns:
            Active carbon-aware trainer
        """
        if region not in self.regions:
            raise ValueError(f"Region {region} not configured")
        
        trainer = CarbonAwareTrainer(
            model=model,
            optimizer=optimizer,
            carbon_model='electricitymap',
            region=region,
            target_carbon_intensity=target_carbon_intensity
        )
        
        await trainer.initialize()
        await trainer.start_training()
        
        self.active_trainers[region] = trainer
        self.current_region = region
        
        logger.info(f"Training started in region {region}")
        return trainer
    
    async def monitor_and_migrate(
        self,
        check_interval_minutes: int = 15
    ) -> None:
        """Monitor carbon intensity and migrate training if beneficial.
        
        Args:
            check_interval_minutes: How often to check for migration opportunities
        """
        if not self.current_region or self.current_region not in self.active_trainers:
            logger.warning("No active training to monitor")
            return
        
        logger.info(f"Starting migration monitoring (check every {check_interval_minutes}m)")
        
        while self.current_region in self.active_trainers:
            try:
                # Get current region's carbon intensity
                current_monitor = self.monitors[self.current_region]
                current_intensity = await current_monitor.get_current_intensity(self.current_region)
                
                if not current_intensity:
                    await asyncio.sleep(check_interval_minutes * 60)
                    continue
                
                # Check if migration is beneficial
                migration_plan = await self._evaluate_migration_opportunity(
                    current_intensity.carbon_intensity
                )
                
                if migration_plan and migration_plan.carbon_benefit_kg >= self.min_migration_benefit_kg:
                    logger.info(f"Migration beneficial: {migration_plan.carbon_benefit_kg:.2f} kg CO2 savings")
                    await self._execute_migration(migration_plan)
                
                await asyncio.sleep(check_interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in migration monitoring: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _evaluate_migration_opportunity(
        self,
        current_carbon_intensity: float
    ) -> Optional[MigrationPlan]:
        """Evaluate if migration to another region is beneficial.
        
        Args:
            current_carbon_intensity: Current region's carbon intensity
            
        Returns:
            Migration plan if beneficial, None otherwise
        """
        if current_carbon_intensity <= self.migration_threshold:
            return None
        
        best_plan = None
        best_benefit = 0
        
        for target_region, region_config in self.regions.items():
            if target_region == self.current_region:
                continue
            
            # Get target region's carbon intensity
            target_monitor = self.monitors.get(target_region)
            if not target_monitor:
                continue
            
            target_intensity = await target_monitor.get_current_intensity(target_region)
            if not target_intensity:
                continue
            
            # Calculate potential carbon savings
            intensity_diff = current_carbon_intensity - target_intensity.carbon_intensity
            if intensity_diff <= 0:
                continue
            
            # Estimate migration parameters
            migration_time_hours = self._estimate_migration_time(
                self.current_region, target_region
            )
            
            # Calculate carbon benefit (simplified)
            remaining_training_hours = 4.0  # Simplified - would track actual remaining time
            carbon_benefit_kg = (intensity_diff / 1000) * 0.4 * remaining_training_hours  # Assume 400W GPU
            
            # Calculate cost impact
            current_cost_per_hour = self.regions[self.current_region].cost_per_gpu_hour
            target_cost_per_hour = region_config.cost_per_gpu_hour
            cost_impact_usd = (target_cost_per_hour - current_cost_per_hour) * remaining_training_hours
            
            # Calculate risk score
            risk_score = self._calculate_migration_risk(target_region, region_config)
            
            if carbon_benefit_kg > best_benefit and risk_score < 0.5:
                best_benefit = carbon_benefit_kg
                best_plan = MigrationPlan(
                    from_region=self.current_region,
                    to_region=target_region,
                    start_time=datetime.now(),
                    estimated_duration_minutes=int(migration_time_hours * 60),
                    carbon_benefit_kg=carbon_benefit_kg,
                    cost_impact_usd=cost_impact_usd,
                    risk_score=risk_score
                )
        
        return best_plan
    
    def _estimate_migration_time(self, from_region: str, to_region: str) -> float:
        """Estimate time required for migration between regions.
        
        Returns:
            Estimated time in hours
        """
        from_config = self.regions[from_region]
        to_config = self.regions[to_region]
        
        # Base migration time includes setup time
        base_time_minutes = to_config.setup_time_minutes
        
        # Add data transfer time (simplified - assume 10GB model)
        model_size_gb = 10.0
        transfer_bandwidth = min(from_config.transfer_bandwidth_gbps, to_config.transfer_bandwidth_gbps)
        transfer_time_minutes = (model_size_gb * 8) / (transfer_bandwidth * 60)  # Convert to minutes
        
        total_time_hours = (base_time_minutes + transfer_time_minutes) / 60
        return total_time_hours
    
    def _calculate_migration_risk(self, target_region: str, region_config: RegionConfig) -> float:
        """Calculate risk score for migration to target region.
        
        Returns:
            Risk score between 0-1 (lower is better)
        """
        risk_factors = []
        
        # Availability risk
        availability_risk = 1.0 - region_config.availability_score
        risk_factors.append(availability_risk * 0.4)
        
        # Historical migration success rate (simplified)
        historical_success_rate = 0.9  # Would track actual success rate
        migration_risk = 1.0 - historical_success_rate
        risk_factors.append(migration_risk * 0.3)
        
        # Setup time risk (longer setup = higher risk)
        setup_risk = min(region_config.setup_time_minutes / 60, 1.0)  # Normalize to 0-1
        risk_factors.append(setup_risk * 0.3)
        
        return sum(risk_factors)
    
    async def _execute_migration(self, plan: MigrationPlan) -> bool:
        """Execute training migration according to plan.
        
        Args:
            plan: Migration plan to execute
            
        Returns:
            True if migration successful, False otherwise
        """
        logger.info(f"Executing migration: {plan.from_region} -> {plan.to_region}")
        
        try:
            # Get current trainer
            current_trainer = self.active_trainers.get(plan.from_region)
            if not current_trainer:
                return False
            
            # 1. Pause current training
            await current_trainer._pause_training("Migration initiated")
            
            # 2. Save checkpoint (simplified - would implement actual checkpointing)
            checkpoint_data = self._create_checkpoint(current_trainer)
            
            # 3. Stop current trainer
            await current_trainer.stop_training()
            await current_trainer.cleanup()
            del self.active_trainers[plan.from_region]
            
            # 4. Start training in target region
            target_trainer = CarbonAwareTrainer(
                model=current_trainer.model,
                optimizer=current_trainer.optimizer,
                carbon_model='electricitymap',
                region=plan.to_region,
                target_carbon_intensity=current_trainer.config.carbon_threshold
            )
            
            await target_trainer.initialize()
            
            # 5. Restore from checkpoint (simplified)
            self._restore_checkpoint(target_trainer, checkpoint_data)
            
            await target_trainer.start_training()
            
            # 6. Update state
            self.active_trainers[plan.to_region] = target_trainer
            self.current_region = plan.to_region
            self.migration_history.append(plan)
            self.total_migrations += 1
            
            logger.info(f"Migration completed successfully to {plan.to_region}")
            return True
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
    
    def _create_checkpoint(self, trainer: CarbonAwareTrainer) -> Dict[str, Any]:
        """Create checkpoint data for migration."""
        return {
            'step': trainer.step,
            'epoch': trainer.epoch,
            'metrics': trainer.metrics,
            'timestamp': datetime.now()
        }
    
    def _restore_checkpoint(self, trainer: CarbonAwareTrainer, checkpoint: Dict[str, Any]) -> None:
        """Restore trainer state from checkpoint."""
        trainer.step = checkpoint['step']
        trainer.epoch = checkpoint['epoch']
        # Note: In real implementation, would restore model state dict
    
    def get_orchestration_metrics(self) -> Dict[str, Any]:
        """Get comprehensive orchestration metrics.
        
        Returns:
            Dictionary with orchestration metrics
        """
        return {
            'current_region': self.current_region,
            'total_regions': len(self.regions),
            'active_trainers': len(self.active_trainers),
            'total_migrations': self.total_migrations,
            'total_carbon_used_kg': self.total_carbon_used_kg,
            'total_cost_used_usd': self.total_cost_used_usd,
            'carbon_budget_remaining_kg': (
                self.carbon_budget_kg - self.total_carbon_used_kg 
                if self.carbon_budget_kg else None
            ),
            'cost_budget_remaining_usd': (
                self.cost_budget_usd - self.total_cost_used_usd
                if self.cost_budget_usd else None
            ),
            'migration_history': len(self.migration_history)
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()