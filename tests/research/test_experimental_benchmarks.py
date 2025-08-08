"""
Tests for experimental benchmarking framework.

This module tests the comprehensive benchmarking capabilities including
baseline comparisons, cross-regional optimization, and statistical validation.
"""

import pytest
import asyncio
import tempfile
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from carbon_aware_trainer.core.types import CarbonIntensity, CarbonForecast
from carbon_aware_trainer.core.advanced_forecasting import AdvancedCarbonForecaster
from carbon_aware_trainer.research.experimental_benchmarks import (
    CarbonForecastBenchmark,
    CrossRegionalOptimizer,
    BenchmarkSuite,
    BaselineModel,
    ResearchMetrics,
    BenchmarkType
)


@pytest.fixture
def sample_test_data():
    """Generate sample carbon intensity data for benchmark testing."""
    data = []
    base_time = datetime.now() - timedelta(hours=200)
    
    for i in range(200):
        timestamp = base_time + timedelta(hours=i)
        # Create realistic carbon intensity pattern
        base_intensity = 120 + 50 * ((i % 24) / 24)  # Daily cycle
        base_intensity += 30 * ((i % 168) / 168)     # Weekly cycle
        base_intensity += (hash(str(i)) % 60) - 30   # Random variation
        
        ci = CarbonIntensity(
            carbon_intensity=max(30, min(350, base_intensity)),
            timestamp=timestamp,
            region="BENCHMARK_REGION",
            renewable_percentage=0.2 + 0.6 * ((i % 24) / 24)
        )
        data.append(ci)
    
    return data


@pytest.fixture
def mock_forecaster():
    """Create mock advanced carbon forecaster."""
    forecaster = Mock(spec=AdvancedCarbonForecaster)
    
    async def mock_transformer_forecast(inputs, horizon_hours):
        # Generate mock predictions
        predictions = []
        base_time = datetime.now()
        for i in range(horizon_hours):
            ci = CarbonIntensity(
                carbon_intensity=100 + 20 * (i % 12) / 12,
                timestamp=base_time + timedelta(hours=i + 1),
                region="MOCK_REGION"
            )
            predictions.append(ci)
        
        # Mock transformer result
        result = Mock()
        result.forecast = Mock()
        result.forecast.data_points = predictions
        result.uncertainty_bounds = [(p.carbon_intensity * 0.9, p.carbon_intensity * 1.1) for p in predictions]
        result.confidence_intervals = [0.8] * len(predictions)
        return result
    
    async def mock_ensemble_forecast(inputs, horizon_hours=48):
        return await mock_transformer_forecast(inputs, horizon_hours)
    
    forecaster.get_transformer_forecast = mock_transformer_forecast
    forecaster.get_ensemble_forecast = mock_ensemble_forecast
    
    return forecaster


class TestCarbonForecastBenchmark:
    """Test cases for carbon forecast benchmarking framework."""
    
    @pytest.fixture
    def benchmark(self):
        """Create benchmark instance with temporary output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark = CarbonForecastBenchmark(temp_dir)
            benchmark.output_dir = Path(temp_dir)  # Ensure it's accessible
            yield benchmark
    
    def test_initialization(self, benchmark):
        """Test benchmark initialization."""
        assert len(benchmark.baselines) > 0
        assert "naive" in benchmark.baselines
        assert "moving_average" in benchmark.baselines
        assert "linear_trend" in benchmark.baselines
        assert "seasonal_naive" in benchmark.baselines
    
    def test_baseline_models(self, benchmark, sample_test_data):
        """Test baseline model predictions."""
        history_data = sample_test_data[:100]
        
        for baseline_name, baseline_model in benchmark.baselines.items():
            predictions = baseline_model.predictor_func(history_data)
            
            assert len(predictions) == 48  # Expected forecast horizon
            assert all(10 <= p <= 800 for p in predictions)  # Reasonable range
            
            # Test specific baseline behaviors
            if baseline_name == "naive":
                # Naive should repeat last value
                expected_value = history_data[-1].carbon_intensity
                assert all(abs(p - expected_value) < 1e-6 for p in predictions)
            
            elif baseline_name == "moving_average":
                # Moving average should be stable
                avg_expected = sum(ci.carbon_intensity for ci in history_data[-24:]) / 24
                assert all(abs(p - avg_expected) < 1e-6 for p in predictions)
    
    @pytest.mark.asyncio
    async def test_calculate_research_metrics(self, benchmark):
        """Test research metrics calculation."""
        predictions = [100, 110, 120, 115, 125, 130]
        actual_values = [105, 115, 118, 120, 130, 135]
        
        metrics = await benchmark._calculate_research_metrics(predictions, actual_values)
        
        assert isinstance(metrics, ResearchMetrics)
        assert metrics.mae > 0
        assert metrics.rmse > 0
        assert metrics.mape >= 0
        assert -1 <= metrics.r2 <= 1
        
        # Test with uncertainty bounds
        from carbon_aware_trainer.core.advanced_forecasting import TransformerForecastResult, AttentionWeights
        from carbon_aware_trainer.core.types import CarbonForecast
        
        # Create mock result with uncertainty bounds
        mock_result = Mock()
        mock_result.uncertainty_bounds = [(p * 0.95, p * 1.05) for p in predictions]
        mock_result.confidence_intervals = [0.9] * len(predictions)
        
        metrics_with_uncertainty = await benchmark._calculate_research_metrics(
            predictions, actual_values, mock_result
        )
        
        assert metrics_with_uncertainty.mean_interval_score >= 0
        assert 0 <= metrics_with_uncertainty.coverage_probability <= 1
        assert metrics_with_uncertainty.sharpness >= 0
    
    def test_aggregate_metrics(self, benchmark):
        """Test metrics aggregation across multiple folds."""
        metrics_list = [
            ResearchMetrics(mae=10.0, rmse=15.0, mape=0.1, r2=0.8),
            ResearchMetrics(mae=12.0, rmse=18.0, mape=0.12, r2=0.75),
            ResearchMetrics(mae=8.0, rmse=12.0, mape=0.08, r2=0.85)
        ]
        
        aggregated = benchmark._aggregate_metrics(metrics_list)
        
        assert isinstance(aggregated, ResearchMetrics)
        assert aggregated.mae == 10.0  # Average of [10, 12, 8]
        assert aggregated.rmse == 15.0  # Average of [15, 18, 12]
        assert aggregated.r2 == 0.8  # Average of [0.8, 0.75, 0.85]
    
    @pytest.mark.asyncio
    async def test_run_accuracy_benchmark(self, benchmark, sample_test_data, mock_forecaster):
        """Test accuracy benchmark execution."""
        # Use subset of data for faster testing
        test_data = sample_test_data[:100]
        
        results = await benchmark.run_accuracy_benchmark(
            test_data,
            mock_forecaster,
            models_to_test=["transformer"],  # Just test transformer
            forecast_horizons=[12, 24]  # Shorter horizons for speed
        )
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Should include transformer and baseline results
        assert any("transformer" in model_name for model_name in results.keys())
        assert any("baseline_" in model_name for model_name in results.keys())
        
        # Each model should have results for each horizon
        for model_name, model_results in results.items():
            if model_results:  # If not empty
                for horizon in ["12h", "24h"]:
                    if horizon in model_results:
                        metrics = model_results[horizon]
                        assert hasattr(metrics, 'mae')
                        assert hasattr(metrics, 'rmse')
    
    @pytest.mark.asyncio
    async def test_save_benchmark_results(self, benchmark):
        """Test benchmark results saving."""
        # Mock results
        mock_results = {
            "model_a": {
                "24h": ResearchMetrics(mae=10.0, rmse=15.0, r2=0.8)
            }
        }
        
        await benchmark._save_benchmark_results("test_benchmark", mock_results)
        
        # Check that files were created
        json_files = list(benchmark.output_dir.glob("test_benchmark_*.json"))
        csv_files = list(benchmark.output_dir.glob("test_benchmark_*.csv"))
        
        assert len(json_files) > 0
        assert len(csv_files) > 0
        
        # Verify JSON content
        with open(json_files[0], 'r') as f:
            saved_data = json.load(f)
            assert "model_a" in saved_data
    
    @pytest.mark.asyncio
    async def test_statistical_tests(self, benchmark):
        """Test statistical significance testing."""
        # Mock results with different performance levels
        results = {
            "model_good": {"24h": ResearchMetrics(mae=8.0)},
            "model_bad": {"24h": ResearchMetrics(mae=15.0)},
            "baseline_naive": {"24h": ResearchMetrics(mae=12.0)}
        }
        
        await benchmark._perform_statistical_tests(results)
        
        assert len(benchmark.statistical_tests) > 0
        
        # Check for specific comparisons
        test_keys = list(benchmark.statistical_tests.keys())
        assert any("model_good_vs_model_bad" in key for key in test_keys)
        
        # Verify statistical test structure
        for test_result in benchmark.statistical_tests.values():
            assert "effect_size" in test_result
            assert "p_value" in test_result
            assert "mae_difference" in test_result


class TestCrossRegionalOptimizer:
    """Test cases for cross-regional optimization."""
    
    @pytest.fixture
    def optimizer(self):
        """Create cross-regional optimizer instance."""
        regions = ["US-CA", "US-WA", "EU-FR", "EU-DE"]
        return CrossRegionalOptimizer(regions)
    
    def test_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert len(optimizer.regions) == 4
        assert "US-CA" in optimizer.regions
        assert "EU-FR" in optimizer.regions
    
    @pytest.mark.asyncio
    async def test_optimize_global_placement(self, optimizer):
        """Test global placement optimization."""
        workload_requirements = {
            "duration_hours": 24,
            "gpu_count": 8,
            "memory_gb": 64
        }
        
        result = await optimizer.optimize_global_placement(workload_requirements, 48)
        
        assert isinstance(result, dict)
        assert "timestamp" in result
        assert "regional_analysis" in result
        assert "recommended_regions" in result
        assert "best_region" in result
        assert "global_carbon_savings" in result
        
        # Verify regional analysis
        regional_analysis = result["regional_analysis"]
        assert len(regional_analysis) == len(optimizer.regions)
        
        for region, analysis in regional_analysis.items():
            assert "optimization_score" in analysis
            assert "expected_carbon_savings" in analysis
            assert "renewable_utilization" in analysis
            assert "recommendation" in analysis
            assert analysis["recommendation"] in ["optimal", "acceptable", "avoid"]
        
        # Best region should be from the available regions
        if result["best_region"]:
            assert result["best_region"] in optimizer.regions
        
        # Recommended regions should be subset of available regions
        for region in result["recommended_regions"]:
            assert region in optimizer.regions


class TestBenchmarkSuite:
    """Test cases for comprehensive benchmark suite."""
    
    @pytest.fixture
    def benchmark_suite(self):
        """Create benchmark suite instance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            suite = BenchmarkSuite(temp_dir)
            yield suite
    
    def test_initialization(self, benchmark_suite):
        """Test benchmark suite initialization."""
        assert benchmark_suite.forecast_benchmark is not None
        assert benchmark_suite.cross_regional_optimizer is not None
        assert len(benchmark_suite.cross_regional_optimizer.regions) == 5
    
    @pytest.mark.asyncio
    async def test_run_comprehensive_evaluation(self, benchmark_suite, sample_test_data, mock_forecaster):
        """Test comprehensive evaluation execution."""
        # Use smaller dataset for faster testing
        test_data = sample_test_data[:50]
        
        results = await benchmark_suite.run_comprehensive_evaluation(test_data, mock_forecaster)
        
        assert isinstance(results, dict)
        assert "evaluation_timestamp" in results
        assert "test_data_size" in results
        assert "benchmarks" in results
        assert "research_summary" in results
        
        # Verify benchmark results structure
        benchmarks = results["benchmarks"]
        assert "accuracy" in benchmarks
        assert "cross_regional" in benchmarks
        
        # Verify research summary
        research_summary = results["research_summary"]
        assert "key_findings" in research_summary
        assert "performance_improvements" in research_summary
        assert "carbon_impact" in research_summary
    
    def test_generate_research_summary(self, benchmark_suite):
        """Test research summary generation."""
        mock_results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "benchmarks": {
                "accuracy": {
                    "transformer": {"24h": ResearchMetrics(mae=8.0)},
                    "baseline_naive": {"24h": ResearchMetrics(mae=12.0)}
                },
                "cross_regional": {
                    "global_carbon_savings": 150.5,
                    "best_region": "EU-NO"
                }
            }
        }
        
        summary = benchmark_suite._generate_research_summary(mock_results)
        
        assert isinstance(summary, dict)
        assert "key_findings" in summary
        assert "performance_improvements" in summary
        assert "carbon_impact" in summary
        
        # Should identify best performing model
        assert len(summary["key_findings"]) > 0
        assert any("transformer" in finding for finding in summary["key_findings"])
    
    def test_make_serializable(self, benchmark_suite):
        """Test serialization of complex objects."""
        complex_obj = {
            "metrics": ResearchMetrics(mae=10.0, rmse=15.0),
            "timestamp": datetime.now(),
            "nested": {"values": [1, 2, 3]}
        }
        
        serialized = benchmark_suite._make_serializable(complex_obj)
        
        assert isinstance(serialized, dict)
        assert "metrics" in serialized
        assert isinstance(serialized["metrics"], dict)  # Should be converted from dataclass
        assert "mae" in serialized["metrics"]
        assert isinstance(serialized["timestamp"], str)  # Should be converted from datetime
    
    def test_generate_markdown_report(self, benchmark_suite):
        """Test markdown report generation."""
        mock_results = {
            "research_summary": {
                "key_findings": [
                    "Transformer model achieved 25% improvement over baseline",
                    "Cross-regional optimization saved 120 kg CO2"
                ]
            },
            "benchmarks": {
                "accuracy": {
                    "transformer": {"24h": ResearchMetrics(mae=8.0, rmse=12.0, r2=0.85, coverage_probability=0.9, sharpness=5.0, carbon_savings_kg=15.0)},
                    "baseline_naive": {"24h": ResearchMetrics(mae=12.0, rmse=18.0, r2=0.7, coverage_probability=0.8, sharpness=8.0, carbon_savings_kg=5.0)}
                },
                "cross_regional": {
                    "best_region": "EU-NO",
                    "global_carbon_savings": 120.5,
                    "recommended_regions": ["EU-NO", "US-WA", "US-CA"]
                }
            }
        }
        
        report = benchmark_suite._generate_markdown_report(mock_results)
        
        assert isinstance(report, str)
        assert "# Carbon-Aware ML Training Research Results" in report
        assert "Executive Summary" in report
        assert "Key Findings" in report
        assert "Methodology" in report
        assert "Results" in report
        
        # Should include key findings
        for finding in mock_results["research_summary"]["key_findings"]:
            assert finding in report
        
        # Should include results table
        assert "| Model |" in report  # Table header
        assert "transformer" in report
        assert "baseline_naive" in report


class TestIntegrationBenchmarks:
    """Integration tests for benchmarking framework."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_benchmark_workflow(self, sample_test_data, mock_forecaster):
        """Test complete benchmark workflow from data to report."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize benchmark suite
            suite = BenchmarkSuite(temp_dir)
            
            # Run comprehensive evaluation
            results = await suite.run_comprehensive_evaluation(
                sample_test_data[:30],  # Smaller dataset for speed
                mock_forecaster
            )
            
            # Verify complete workflow
            assert results is not None
            assert "benchmarks" in results
            assert "research_summary" in results
            
            # Check that files were saved
            output_files = list(Path(temp_dir).glob("comprehensive_benchmark_*.json"))
            assert len(output_files) > 0
            
            # Check markdown report generation
            summary_files = list(Path(temp_dir).glob("research_summary_*.md"))
            assert len(summary_files) > 0
            
            # Verify report content
            with open(summary_files[0], 'r') as f:
                report_content = f.read()
                assert "Carbon-Aware ML Training Research Results" in report_content
    
    @pytest.mark.asyncio
    async def test_benchmark_performance_metrics(self, sample_test_data):
        """Test benchmark performance and timing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark = CarbonForecastBenchmark(temp_dir)
            
            # Measure baseline model performance
            start_time = datetime.now()
            
            for baseline_name, baseline_model in benchmark.baselines.items():
                predictions = baseline_model.predictor_func(sample_test_data[:50])
                assert len(predictions) == 48
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Baseline models should be fast (< 1 second for small dataset)
            assert execution_time < 1.0
    
    @pytest.mark.asyncio
    async def test_benchmark_error_handling(self, mock_forecaster):
        """Test benchmark error handling with invalid data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark = CarbonForecastBenchmark(temp_dir)
            
            # Test with empty data
            empty_data = []
            
            try:
                results = await benchmark.run_accuracy_benchmark(
                    empty_data, mock_forecaster, models_to_test=["transformer"]
                )
                # Should handle gracefully
                assert isinstance(results, dict)
            except Exception as e:
                # Or should raise appropriate exception
                assert "data" in str(e).lower()
            
            # Test with insufficient data
            minimal_data = [
                CarbonIntensity(100, datetime.now(), "TEST", None)
            ]
            
            results = await benchmark.run_accuracy_benchmark(
                minimal_data, mock_forecaster, models_to_test=["transformer"]
            )
            
            # Should handle minimal data gracefully
            assert isinstance(results, dict)