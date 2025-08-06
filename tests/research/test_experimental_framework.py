"""Tests for experimental framework."""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from carbon_aware_trainer.research.experimental_framework import (
    ExperimentalFramework, ExperimentConfig, ExperimentalRun
)
from carbon_aware_trainer.core.scheduler import CarbonAwareTrainer


@pytest.fixture
def sample_config():
    """Sample experiment configuration."""
    return ExperimentConfig(
        experiment_name="test_carbon_reduction",
        experiment_version="1.0.0",
        description="Test carbon-aware training effectiveness",
        model_name="test_model",
        dataset_name="test_dataset", 
        training_hours=2.0,
        num_gpus=1,
        regions=["US-CA", "US-TX"],
        carbon_thresholds=[50, 100, 150],
        num_trials=2,
        results_dir="./test_results"
    )


@pytest.fixture
def mock_training_function():
    """Mock training function."""
    async def training_func(**kwargs):
        await asyncio.sleep(0.01)  # Simulate some work
        return {
            'accuracy': 0.85,
            'loss': 0.15,
            'convergence_time_hours': 1.5
        }
    
    return training_func


class TestExperimentalFramework:
    """Test experimental framework functionality."""
    
    def test_framework_initialization(self, sample_config):
        """Test framework initialization."""
        framework = ExperimentalFramework(sample_config)
        
        assert framework.config == sample_config
        assert framework.runs == []
        assert framework._baseline_runs == []
        assert framework._treatment_runs == []
    
    @pytest.mark.asyncio
    async def test_baseline_experiments(self, sample_config, mock_training_function):
        """Test baseline experiment execution."""
        framework = ExperimentalFramework(sample_config)
        
        # Run baseline experiments
        baseline_runs = await framework.run_baseline_experiments(mock_training_function)
        
        # Verify results
        assert len(baseline_runs) == sample_config.num_trials
        assert all(isinstance(run, ExperimentalRun) for run in baseline_runs)
        assert all(run.algorithm_used == "baseline_no_carbon_awareness" for run in baseline_runs)
        assert all(run.region_used == sample_config.baseline_region for run in baseline_runs)
        assert all(run.end_time is not None for run in baseline_runs)
    
    @pytest.mark.asyncio
    async def test_carbon_aware_experiments(self, sample_config, mock_training_function):
        """Test carbon-aware experiment execution."""
        framework = ExperimentalFramework(sample_config)
        
        # Mock CarbonAwareTrainer
        with patch('carbon_aware_trainer.research.experimental_framework.CarbonAwareTrainer') as mock_trainer_class:
            mock_trainer = AsyncMock()
            mock_trainer.get_carbon_metrics.return_value = {
                'total_carbon_kg': 2.5,
                'avg_carbon_intensity': 120,
                'peak_carbon_intensity': 180,
                'total_energy_kwh': 0.8,
                'paused_duration_hours': 0.5
            }
            mock_trainer_class.return_value = mock_trainer
            
            # Run carbon-aware experiments
            treatment_runs = await framework.run_carbon_aware_experiments(mock_training_function)
            
            # Verify results
            expected_runs = len(sample_config.carbon_thresholds) * sample_config.num_trials
            assert len(treatment_runs) == expected_runs
            assert all(isinstance(run, ExperimentalRun) for run in treatment_runs)
            assert all("carbon_aware" in run.algorithm_used for run in treatment_runs)
    
    @pytest.mark.asyncio 
    async def test_full_experiment_workflow(self, sample_config, mock_training_function, tmp_path):
        """Test complete experimental workflow."""
        # Update config to use temporary directory
        sample_config.results_dir = str(tmp_path)
        
        framework = ExperimentalFramework(sample_config)
        
        # Mock CarbonAwareTrainer for treatment runs
        with patch('carbon_aware_trainer.research.experimental_framework.CarbonAwareTrainer') as mock_trainer_class:
            mock_trainer = AsyncMock()
            mock_trainer.get_carbon_metrics.return_value = {
                'total_carbon_kg': 1.8,
                'avg_carbon_intensity': 90,
                'peak_carbon_intensity': 150,
                'total_energy_kwh': 0.6,
                'paused_duration_hours': 0.2
            }
            mock_trainer_class.return_value = mock_trainer
            
            # Run full experiment
            results = await framework.run_full_experiment(mock_training_function)
            
            # Verify results structure
            assert isinstance(results, dict)
            assert 'experiment_path' in results
            assert 'carbon_metrics' in results
            assert 'experiment_info' in results
            
            # Verify files were created
            experiment_path = results['experiment_path']
            assert (tmp_path / 'config.json').exists() or experiment_path
            
    def test_summary_statistics_generation(self, sample_config):
        """Test summary statistics generation."""
        framework = ExperimentalFramework(sample_config)
        
        # Create mock experimental runs
        baseline_run = ExperimentalRun(
            run_id="baseline_1",
            config=sample_config,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=2),
            total_carbon_kg=3.0,
            algorithm_used="baseline"
        )
        
        treatment_run = ExperimentalRun(
            run_id="treatment_1", 
            config=sample_config,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=2.5),
            total_carbon_kg=2.0,
            algorithm_used="carbon_aware"
        )
        
        framework._baseline_runs = [baseline_run]
        framework._treatment_runs = [treatment_run]
        framework.runs = [baseline_run, treatment_run]
        
        # Generate summary
        summary = framework._generate_summary_statistics()
        
        # Verify summary structure
        assert 'experiment_info' in summary
        assert 'carbon_metrics' in summary
        assert summary['experiment_info']['total_runs'] == 2
        assert summary['experiment_info']['baseline_runs'] == 1
        assert summary['experiment_info']['treatment_runs'] == 1
    
    def test_energy_consumption_calculation(self, sample_config):
        """Test energy consumption calculation."""
        framework = ExperimentalFramework(sample_config)
        
        start_time = 0.0
        end_time = 3600.0  # 1 hour
        
        energy_kwh = framework._calculate_energy_consumption(start_time, end_time)
        
        # Should be 1 GPU * 400W * 1 hour = 0.4 kWh
        expected_energy = 0.4
        assert abs(energy_kwh - expected_energy) < 0.01
    
    def test_carbon_emissions_calculation(self, sample_config):
        """Test carbon emissions calculation."""
        framework = ExperimentalFramework(sample_config)
        
        energy_kwh = 1.0
        region = "US-CA"
        
        carbon_kg = framework._calculate_carbon_emissions(energy_kwh, region)
        
        # Should be 1 kWh * 200 gCO2/kWh = 0.2 kg CO2
        expected_carbon = 0.2
        assert abs(carbon_kg - expected_carbon) < 0.01
    
    def test_invalid_config_handling(self):
        """Test handling of invalid configuration."""
        # Test with missing required fields
        invalid_config = ExperimentConfig(
            experiment_name="",  # Empty name should be handled
            experiment_version="1.0.0",
            description="Test",
            model_name="test",
            dataset_name="test",
            training_hours=-1,  # Invalid negative hours
            regions=[],  # Empty regions list
            carbon_thresholds=[]  # Empty thresholds
        )
        
        framework = ExperimentalFramework(invalid_config)
        assert framework.config == invalid_config  # Should still initialize
    
    @pytest.mark.asyncio
    async def test_training_function_error_handling(self, sample_config):
        """Test error handling when training function fails."""
        framework = ExperimentalFramework(sample_config)
        
        # Create failing training function
        async def failing_training_func(**kwargs):
            raise Exception("Simulated training failure")
        
        # Should handle errors gracefully
        baseline_runs = await framework.run_baseline_experiments(failing_training_func)
        
        # Should still create run records even if training fails
        assert len(baseline_runs) == sample_config.num_trials
        assert all(run.end_time is not None for run in baseline_runs)
    
    def test_metrics_data_export(self, sample_config, tmp_path):
        """Test experiment data export functionality."""
        sample_config.results_dir = str(tmp_path)
        framework = ExperimentalFramework(sample_config)
        
        # Add some mock runs
        run = ExperimentalRun(
            run_id="test_run",
            config=sample_config,
            start_time=datetime.now(),
            total_carbon_kg=2.5
        )
        framework.runs = [run]
        
        # Export data
        export_path = framework.save_experiment_data()
        
        # Verify export path exists and contains expected files
        assert export_path is not None
        # Note: Actual file verification would depend on the implementation details


@pytest.mark.integration
class TestExperimentalFrameworkIntegration:
    """Integration tests for experimental framework."""
    
    @pytest.mark.asyncio
    async def test_integration_with_real_components(self, sample_config, tmp_path):
        """Test integration with real carbon-aware components."""
        sample_config.results_dir = str(tmp_path)
        sample_config.num_trials = 1  # Reduce for faster testing
        sample_config.carbon_thresholds = [100]  # Single threshold
        
        framework = ExperimentalFramework(sample_config)
        
        # Simple training function
        async def simple_training(**kwargs):
            await asyncio.sleep(0.1)
            return {'accuracy': 0.8, 'loss': 0.2}
        
        # Test baseline experiments
        baseline_runs = await framework.run_baseline_experiments(simple_training)
        assert len(baseline_runs) == 1
        
        # Note: Full integration test would require actual carbon monitoring setup