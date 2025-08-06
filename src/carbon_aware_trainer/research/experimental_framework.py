"""Experimental framework for carbon-aware training research."""

import asyncio
import json
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Union
import logging

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from ..core.scheduler import CarbonAwareTrainer
from ..core.types import CarbonIntensity, TrainingMetrics
from ..monitoring.metrics import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for carbon-aware training experiments."""
    
    # Experiment identification
    experiment_name: str
    experiment_version: str
    description: str
    
    # Training parameters
    model_name: str
    dataset_name: str
    training_hours: float
    num_gpus: int = 1
    avg_gpu_power_watts: float = 400
    
    # Carbon parameters
    regions: List[str]
    carbon_thresholds: List[float]
    baseline_region: str = "US-CA"
    
    # Research parameters
    num_trials: int = 3
    statistical_significance: float = 0.05
    confidence_interval: float = 0.95
    
    # Output configuration
    results_dir: str = "./experiments"
    save_raw_data: bool = True
    generate_plots: bool = True


@dataclass
class ExperimentalRun:
    """Single experimental run data."""
    
    run_id: str
    config: ExperimentConfig
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Carbon metrics
    total_carbon_kg: float = 0.0
    avg_carbon_intensity: float = 0.0
    peak_carbon_intensity: float = 0.0
    carbon_savings_kg: float = 0.0
    
    # Training metrics
    total_energy_kwh: float = 0.0
    training_efficiency: float = 0.0
    pause_duration_hours: float = 0.0
    
    # Performance metrics
    model_accuracy: Optional[float] = None
    training_loss: Optional[float] = None
    convergence_time_hours: Optional[float] = None
    
    # Additional metadata
    region_used: str = ""
    algorithm_used: str = ""
    raw_data_path: Optional[str] = None


class ExperimentalFramework:
    """Research framework for systematic carbon-aware training experiments."""
    
    def __init__(self, config: ExperimentConfig):
        """Initialize experimental framework.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_collector = MetricsCollector()
        self.runs: List[ExperimentalRun] = []
        
        # Experiment state
        self._current_run: Optional[ExperimentalRun] = None
        self._baseline_runs: List[ExperimentalRun] = []
        self._treatment_runs: List[ExperimentalRun] = []
    
    async def run_baseline_experiments(
        self,
        training_function: Callable,
        **training_kwargs
    ) -> List[ExperimentalRun]:
        """Run baseline experiments without carbon awareness.
        
        Args:
            training_function: Function to train model
            **training_kwargs: Additional training arguments
            
        Returns:
            List of baseline experimental runs
        """
        logger.info(f"Starting baseline experiments: {self.config.num_trials} runs")
        
        baseline_runs = []
        
        for trial in range(self.config.num_trials):
            run_id = f"baseline_{trial+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            run = ExperimentalRun(
                run_id=run_id,
                config=self.config,
                start_time=datetime.now(),
                region_used=self.config.baseline_region,
                algorithm_used="baseline_no_carbon_awareness"
            )
            
            logger.info(f"Running baseline trial {trial+1}/{self.config.num_trials}")
            
            # Run without carbon awareness
            start_time = time.time()
            
            try:
                # Simulate or run actual training
                result = await self._execute_training_run(
                    training_function,
                    carbon_aware=False,
                    region=self.config.baseline_region,
                    **training_kwargs
                )
                
                # Update run with results
                run.end_time = datetime.now()
                run.total_energy_kwh = self._calculate_energy_consumption(start_time, time.time())
                run.total_carbon_kg = self._calculate_carbon_emissions(
                    run.total_energy_kwh,
                    self.config.baseline_region
                )
                
                if isinstance(result, dict):
                    run.model_accuracy = result.get('accuracy')
                    run.training_loss = result.get('loss')
                    run.convergence_time_hours = result.get('convergence_time_hours')
                
                baseline_runs.append(run)
                self.runs.append(run)
                
            except Exception as e:
                logger.error(f"Baseline trial {trial+1} failed: {e}")
                run.end_time = datetime.now()
                baseline_runs.append(run)
        
        self._baseline_runs = baseline_runs
        logger.info(f"Completed baseline experiments: {len(baseline_runs)} runs")
        return baseline_runs
    
    async def run_carbon_aware_experiments(
        self,
        training_function: Callable,
        **training_kwargs
    ) -> List[ExperimentalRun]:
        """Run carbon-aware training experiments.
        
        Args:
            training_function: Function to train model
            **training_kwargs: Additional training arguments
            
        Returns:
            List of carbon-aware experimental runs
        """
        logger.info(f"Starting carbon-aware experiments: {self.config.num_trials} runs")
        
        treatment_runs = []
        
        for threshold in self.config.carbon_thresholds:
            for trial in range(self.config.num_trials):
                run_id = f"carbon_aware_t{threshold}_{trial+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                run = ExperimentalRun(
                    run_id=run_id,
                    config=self.config,
                    start_time=datetime.now(),
                    algorithm_used=f"carbon_aware_threshold_{threshold}"
                )
                
                logger.info(f"Running carbon-aware trial {trial+1}/{self.config.num_trials} (threshold: {threshold})")
                
                try:
                    # Initialize carbon-aware trainer
                    trainer = CarbonAwareTrainer(
                        carbon_model='electricitymap',
                        region=self.config.regions[0],  # Start with first region
                        target_carbon_intensity=threshold
                    )
                    
                    start_time = time.time()
                    
                    async with trainer:
                        await trainer.start_training()
                        
                        # Execute training with carbon awareness
                        result = await self._execute_carbon_aware_training(
                            trainer,
                            training_function,
                            **training_kwargs
                        )
                        
                        await trainer.stop_training()
                    
                    # Collect metrics from trainer
                    carbon_metrics = trainer.get_carbon_metrics()
                    
                    run.end_time = datetime.now()
                    run.total_carbon_kg = carbon_metrics['total_carbon_kg']
                    run.avg_carbon_intensity = carbon_metrics['avg_carbon_intensity']
                    run.peak_carbon_intensity = carbon_metrics['peak_carbon_intensity']
                    run.total_energy_kwh = carbon_metrics['total_energy_kwh']
                    run.pause_duration_hours = carbon_metrics['paused_duration_hours']
                    run.region_used = self._determine_primary_region(trainer)
                    
                    # Calculate carbon savings vs baseline
                    if self._baseline_runs:
                        baseline_avg_carbon = np.mean([r.total_carbon_kg for r in self._baseline_runs])
                        run.carbon_savings_kg = baseline_avg_carbon - run.total_carbon_kg
                    
                    if isinstance(result, dict):
                        run.model_accuracy = result.get('accuracy')
                        run.training_loss = result.get('loss')
                        run.convergence_time_hours = result.get('convergence_time_hours')
                    
                    treatment_runs.append(run)
                    self.runs.append(run)
                    
                except Exception as e:
                    logger.error(f"Carbon-aware trial failed: {e}")
                    run.end_time = datetime.now()
                    treatment_runs.append(run)
        
        self._treatment_runs = treatment_runs
        logger.info(f"Completed carbon-aware experiments: {len(treatment_runs)} runs")
        return treatment_runs
    
    async def _execute_training_run(
        self,
        training_function: Callable,
        carbon_aware: bool = False,
        region: str = "US-CA",
        **kwargs
    ) -> Any:
        """Execute a single training run."""
        # This would integrate with actual ML training frameworks
        # For now, simulate training with realistic metrics
        
        await asyncio.sleep(0.1)  # Simulate training time
        
        # Simulate training results
        return {
            'accuracy': 0.85 + np.random.normal(0, 0.02) if HAS_NUMPY else 0.85,
            'loss': 0.15 + np.random.normal(0, 0.01) if HAS_NUMPY else 0.15,
            'convergence_time_hours': self.config.training_hours * (0.9 + np.random.uniform(0, 0.2)) if HAS_NUMPY else self.config.training_hours
        }
    
    async def _execute_carbon_aware_training(
        self,
        trainer: CarbonAwareTrainer,
        training_function: Callable,
        **kwargs
    ) -> Any:
        """Execute carbon-aware training with proper integration."""
        # This would integrate trainer with actual training loop
        # For demonstration, simulate the integration
        
        total_steps = 1000  # Simulate total training steps
        
        for step in range(total_steps):
            # Simulate training batch
            batch_data = f"batch_{step}"
            
            try:
                # Execute carbon-aware training step
                await trainer.train_step(batch_data)
                
            except StopIteration:
                # Training stopped due to carbon constraints
                logger.info(f"Training stopped at step {step} due to carbon constraints")
                break
            
            # Simulate occasional progress check
            if step % 100 == 0:
                await asyncio.sleep(0.01)  # Simulate computation time
        
        # Return simulated results
        return await self._execute_training_run(training_function, carbon_aware=True, **kwargs)
    
    def _calculate_energy_consumption(self, start_time: float, end_time: float) -> float:
        """Calculate energy consumption based on training duration."""
        duration_hours = (end_time - start_time) / 3600
        power_kw = (self.config.avg_gpu_power_watts * self.config.num_gpus) / 1000
        return power_kw * duration_hours
    
    def _calculate_carbon_emissions(self, energy_kwh: float, region: str) -> float:
        """Calculate carbon emissions based on energy and region."""
        # Simplified regional carbon intensities (gCO2/kWh)
        regional_intensities = {
            'US-CA': 200,  # California (cleaner grid)
            'US-TX': 450,  # Texas (coal/gas heavy)
            'EU-FR': 80,   # France (nuclear heavy)
            'EU-DE': 350,  # Germany (mixed)
            'CN-GD': 550,  # China Guangdong (coal heavy)
        }
        
        intensity = regional_intensities.get(region, 300)  # Default intensity
        return energy_kwh * (intensity / 1000)  # Convert to kg CO2
    
    def _determine_primary_region(self, trainer: CarbonAwareTrainer) -> str:
        """Determine primary region used during training."""
        return trainer.region  # Simplified - would track actual usage
    
    def save_experiment_data(self) -> str:
        """Save all experimental data to files.
        
        Returns:
            Path to saved experiment directory
        """
        experiment_dir = self.results_dir / f"{self.config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(experiment_dir / "config.json", 'w') as f:
            json.dump(asdict(self.config), f, indent=2, default=str)
        
        # Save all runs data
        runs_data = [asdict(run) for run in self.runs]
        with open(experiment_dir / "runs.json", 'w') as f:
            json.dump(runs_data, f, indent=2, default=str)
        
        # Save summary statistics
        summary = self._generate_summary_statistics()
        with open(experiment_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Experiment data saved to {experiment_dir}")
        return str(experiment_dir)
    
    def _generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics from all runs."""
        if not self.runs:
            return {}
        
        baseline_carbon = [r.total_carbon_kg for r in self._baseline_runs if r.total_carbon_kg > 0]
        treatment_carbon = [r.total_carbon_kg for r in self._treatment_runs if r.total_carbon_kg > 0]
        
        summary = {
            "experiment_info": {
                "name": self.config.experiment_name,
                "total_runs": len(self.runs),
                "baseline_runs": len(self._baseline_runs),
                "treatment_runs": len(self._treatment_runs)
            },
            "carbon_metrics": {},
            "performance_metrics": {}
        }
        
        if baseline_carbon and treatment_carbon:
            if HAS_NUMPY:
                baseline_mean = np.mean(baseline_carbon)
                treatment_mean = np.mean(treatment_carbon)
                carbon_reduction_pct = ((baseline_mean - treatment_mean) / baseline_mean) * 100
                
                summary["carbon_metrics"] = {
                    "baseline_mean_kg": baseline_mean,
                    "baseline_std_kg": np.std(baseline_carbon),
                    "treatment_mean_kg": treatment_mean, 
                    "treatment_std_kg": np.std(treatment_carbon),
                    "carbon_reduction_kg": baseline_mean - treatment_mean,
                    "carbon_reduction_percent": carbon_reduction_pct
                }
            else:
                baseline_mean = sum(baseline_carbon) / len(baseline_carbon)
                treatment_mean = sum(treatment_carbon) / len(treatment_carbon)
                carbon_reduction_pct = ((baseline_mean - treatment_mean) / baseline_mean) * 100
                
                summary["carbon_metrics"] = {
                    "baseline_mean_kg": baseline_mean,
                    "treatment_mean_kg": treatment_mean,
                    "carbon_reduction_kg": baseline_mean - treatment_mean,
                    "carbon_reduction_percent": carbon_reduction_pct
                }
        
        return summary
    
    async def run_full_experiment(
        self,
        training_function: Callable,
        **training_kwargs
    ) -> Dict[str, Any]:
        """Run complete experimental study with baseline and treatment.
        
        Args:
            training_function: Function to train model
            **training_kwargs: Additional training arguments
            
        Returns:
            Complete experimental results
        """
        logger.info(f"Starting full experiment: {self.config.experiment_name}")
        
        # Run baseline experiments
        baseline_results = await self.run_baseline_experiments(training_function, **training_kwargs)
        
        # Run carbon-aware experiments  
        treatment_results = await self.run_carbon_aware_experiments(training_function, **training_kwargs)
        
        # Save all data
        experiment_path = self.save_experiment_data()
        
        # Generate final summary
        summary = self._generate_summary_statistics()
        summary["experiment_path"] = experiment_path
        
        logger.info(f"Experiment completed. Results saved to {experiment_path}")
        
        return summary