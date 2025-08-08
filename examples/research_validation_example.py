#!/usr/bin/env python3
"""
Research Validation Example for Carbon-Aware Training.

This example demonstrates how to use the advanced research validation framework
to conduct rigorous academic-quality evaluations of carbon forecasting models
with statistical significance testing.
"""

import asyncio
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from carbon_aware_trainer.core.types import CarbonIntensity, CarbonForecast
from carbon_aware_trainer.core.monitor import CarbonMonitor
from carbon_aware_trainer.core.advanced_forecasting import (
    AdvancedCarbonForecaster, 
    MultiModalInputs,
    ForecastModel
)
from carbon_aware_trainer.research.experimental_benchmarks import (
    BenchmarkSuite,
    CarbonForecastBenchmark,
    CrossRegionalOptimizer
)
from carbon_aware_trainer.research.comparative_analysis import (
    BaselineComparator,
    StatisticalValidator,
    PerformanceAnalyzer,
    StatisticalTest
)


def setup_logging():
    """Setup logging for the example."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def generate_synthetic_carbon_data(hours: int = 2000, region: str = "SYNTHETIC") -> list[CarbonIntensity]:
    """Generate synthetic carbon intensity data for validation."""
    print(f"📊 Generating {hours} hours of synthetic carbon data for {region}")
    
    data = []
    base_time = datetime.now() - timedelta(hours=hours)
    
    for i in range(hours):
        timestamp = base_time + timedelta(hours=i)
        
        # Create realistic patterns
        # Daily cycle (peak during day, low at night)
        daily_factor = 0.8 + 0.4 * ((i % 24) / 24)
        if 6 <= (i % 24) <= 20:  # Daytime
            daily_factor += 0.3
        
        # Weekly cycle (lower on weekends)
        weekly_factor = 1.0
        day_of_week = (i // 24) % 7
        if day_of_week in [5, 6]:  # Weekend
            weekly_factor = 0.85
        
        # Seasonal trend (higher in winter/summer)
        month = ((i // (24 * 30)) % 12) + 1
        seasonal_factor = 1.0 + 0.2 * (abs(month - 6.5) / 6.5)
        
        # Random variation
        import random
        random.seed(42 + i)  # Reproducible randomness
        noise_factor = 1.0 + random.gauss(0, 0.15)
        
        # Base carbon intensity with all factors
        base_intensity = 120 * daily_factor * weekly_factor * seasonal_factor * noise_factor
        
        # Renewable percentage (inversely correlated with carbon intensity)
        renewable_pct = max(0.1, min(0.8, 0.6 - (base_intensity - 100) / 400))
        renewable_pct += random.gauss(0, 0.05)  # Add noise
        renewable_pct = max(0.0, min(1.0, renewable_pct))
        
        ci = CarbonIntensity(
            carbon_intensity=max(30, min(400, base_intensity)),
            timestamp=timestamp,
            region=region,
            renewable_percentage=renewable_pct
        )
        data.append(ci)
    
    avg_intensity = sum(ci.carbon_intensity for ci in data) / len(data)
    avg_renewable = sum(ci.renewable_percentage or 0 for ci in data) / len(data)
    
    print(f"   ✅ Generated data: avg carbon = {avg_intensity:.1f} gCO2/kWh, avg renewable = {avg_renewable:.1%}")
    return data


async def create_mock_forecaster(region: str) -> AdvancedCarbonForecaster:
    """Create a mock forecaster for testing."""
    class MockMonitor:
        async def get_forecast(self, region: str, hours: int = 48):
            # Generate simple forecast
            data_points = []
            base_time = datetime.now()
            for i in range(hours):
                ci = CarbonIntensity(
                    carbon_intensity=100 + 30 * (i % 12) / 12,  # Simple pattern
                    timestamp=base_time + timedelta(hours=i + 1),
                    region=region
                )
                data_points.append(ci)
            
            return CarbonForecast(
                region=region,
                forecast_time=base_time,
                data_points=data_points
            )
    
    mock_monitor = MockMonitor()
    return AdvancedCarbonForecaster(mock_monitor, region)


async def run_benchmark_validation():
    """Run comprehensive benchmark validation."""
    print("\n🧪 RUNNING BENCHMARK VALIDATION")
    print("=" * 50)
    
    # Generate test data
    test_data = generate_synthetic_carbon_data(hours=500, region="BENCHMARK_REGION")
    
    # Create forecaster
    forecaster = await create_mock_forecaster("BENCHMARK_REGION")
    
    # Initialize benchmark suite
    output_dir = "./research_validation_results"
    Path(output_dir).mkdir(exist_ok=True)
    
    benchmark_suite = BenchmarkSuite(output_dir)
    
    print(f"\n📈 Running comprehensive evaluation with {len(test_data)} data points")
    
    # Run comprehensive evaluation
    results = await benchmark_suite.run_comprehensive_evaluation(test_data, forecaster)
    
    print(f"✅ Evaluation completed!")
    print(f"   - Total benchmarks: {len(results.get('benchmarks', {}))}")
    print(f"   - Key findings: {len(results.get('research_summary', {}).get('key_findings', []))}")
    
    # Print key findings
    if 'research_summary' in results and 'key_findings' in results['research_summary']:
        print(f"\n🔍 Key Research Findings:")
        for i, finding in enumerate(results['research_summary']['key_findings'], 1):
            print(f"   {i}. {finding}")
    
    # Print performance improvements
    if 'research_summary' in results and 'performance_improvements' in results['research_summary']:
        improvements = results['research_summary']['performance_improvements']
        print(f"\n📊 Performance Improvements:")
        for metric, improvement in improvements.items():
            if isinstance(improvement, dict) and 'improvement_percent' in improvement:
                print(f"   - {metric}: {improvement['improvement_percent']:.1f}% improvement")
    
    return results


async def run_statistical_validation():
    """Run statistical validation tests."""
    print("\n📊 RUNNING STATISTICAL VALIDATION")  
    print("=" * 50)
    
    # Generate test data for statistical validation
    test_data = generate_synthetic_carbon_data(hours=200, region="STATS_REGION")
    
    # Create statistical validator
    validator = StatisticalValidator(random_seed=42)
    
    # Mock model function for testing
    async def mock_model_function(training_data, forecast_horizon):
        """Simple mock model that predicts average of recent values."""
        if len(training_data) < 24:
            return [150.0] * forecast_horizon  # Default prediction
        
        recent_values = [ci.carbon_intensity for ci in training_data[-24:]]
        avg_prediction = sum(recent_values) / len(recent_values)
        
        # Add slight trend
        predictions = []
        for i in range(forecast_horizon):
            trend_factor = 1.0 + (i * 0.01)  # 1% increase per hour
            pred = avg_prediction * trend_factor
            predictions.append(max(50, min(300, pred)))
        
        return predictions
    
    print("🔄 Running temporal cross-validation...")
    cv_results = await validator.cross_validate_temporal(
        test_data, mock_model_function, n_folds=5, min_train_size=100
    )
    
    print(f"✅ Cross-validation completed!")
    print(f"   - Successful folds: {cv_results['n_successful_folds']}")
    print(f"   - Average MAE: {cv_results['mae_mean']:.2f} ± {cv_results['mae_std']:.2f}")
    print(f"   - Average R²: {cv_results['r2_mean']:.3f} ± {cv_results['r2_std']:.3f}")
    
    print("🛡️ Running robustness validation...")
    robustness_results = await validator.validate_model_robustness(
        mock_model_function, test_data, perturbation_levels=[0.05, 0.1, 0.2]
    )
    
    print(f"✅ Robustness validation completed!")
    print(f"   - Robustness score: {robustness_results['robustness_score']:.3f}")
    print(f"   - Noise levels tested: {len(robustness_results['perturbation_analysis'])}")
    
    return cv_results, robustness_results


async def run_baseline_comparison():
    """Run baseline model comparison."""
    print("\n⚖️ RUNNING BASELINE COMPARISON")
    print("=" * 50)
    
    # Create baseline comparator
    comparator = BaselineComparator(significance_level=0.05, minimum_effect_size=0.1)
    
    # Simulate model results for comparison
    print("🎯 Simulating model performance data...")
    
    # Simulate different model performance levels
    import random
    random.seed(42)
    
    # Advanced model results (better performance)
    advanced_model_results = [random.gauss(8.0, 1.5) for _ in range(50)]  # Lower MAE is better
    
    # Baseline model results (worse performance)  
    baseline_model_results = [random.gauss(12.0, 2.0) for _ in range(50)]
    
    # Another advanced model for comparison
    transformer_model_results = [random.gauss(7.5, 1.2) for _ in range(50)]
    
    print("📊 Conducting pairwise statistical comparisons...")
    
    # Compare advanced vs baseline
    comparison1 = await comparator.compare_models(
        advanced_model_results,
        baseline_model_results,
        "Advanced Carbon Forecaster",
        "Baseline (Moving Average)",
        "MAE (gCO2/kWh)",
        StatisticalTest.PAIRED_T_TEST
    )
    
    print(f"✅ Advanced vs Baseline:")
    print(f"   - Improvement: {comparison1.improvement_percent:.1f}%")
    print(f"   - Statistical significance: p = {comparison1.statistical_test.p_value:.4f}")
    print(f"   - Effect size: {comparison1.statistical_test.effect_size:.3f}")
    print(f"   - Practically significant: {comparison1.practical_significance}")
    
    # Compare transformer vs advanced
    comparison2 = await comparator.compare_models(
        transformer_model_results,
        advanced_model_results,
        "Transformer Forecaster", 
        "Advanced Carbon Forecaster",
        "MAE (gCO2/kWh)",
        StatisticalTest.BOOTSTRAP
    )
    
    print(f"✅ Transformer vs Advanced:")
    print(f"   - Improvement: {comparison2.improvement_percent:.1f}%")
    print(f"   - Statistical significance: p = {comparison2.statistical_test.p_value:.4f}")
    print(f"   - Bootstrap confidence: {comparison2.statistical_test.confidence_interval}")
    print(f"   - Practically significant: {comparison2.practical_significance}")
    
    # Generate comparison summary
    summary = comparator.generate_comparison_summary()
    print(f"\n📋 Comparison Summary:")
    print(f"   - Total comparisons: {summary['total_comparisons']}")
    print(f"   - Statistically significant: {summary['significant_improvements']}")
    print(f"   - Practically significant: {summary['practically_significant']}")
    
    return summary


async def run_performance_analysis():
    """Run performance analysis of forecasting models."""
    print("\n⚡ RUNNING PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    # Create performance analyzer
    analyzer = PerformanceAnalyzer()
    
    # Mock model function for performance testing
    async def performance_test_model(test_data, forecast_horizon):
        """Mock model with realistic computation time."""
        # Simulate computation time based on input size
        import time
        computation_time = len(test_data) * 0.0001  # Realistic scaling
        await asyncio.sleep(computation_time)
        
        # Generate predictions
        if len(test_data) > 0:
            avg_intensity = sum(ci.carbon_intensity for ci in test_data[-24:]) / min(24, len(test_data))
        else:
            avg_intensity = 150.0
        
        return [avg_intensity + (i * 0.5) for i in range(forecast_horizon)]
    
    print("🔍 Profiling model performance across different scales...")
    
    # Run performance profiling
    performance_results = await analyzer.profile_model_performance(
        performance_test_model,
        test_sizes=[50, 100, 300, 500],  # Different input sizes
        forecast_horizons=[12, 24, 48]   # Different forecast horizons
    )
    
    print(f"✅ Performance analysis completed!")
    
    # Display size scaling results
    if performance_results['size_scaling']:
        print(f"\n📏 Input Size Scaling:")
        for result in performance_results['size_scaling']:
            print(f"   - Size {result['input_size']:3d}: {result['processing_time_seconds']:.3f}s "
                  f"({result['throughput_points_per_second']:.0f} points/s)")
    
    # Display horizon scaling results
    if performance_results['horizon_scaling']:
        print(f"\n🔭 Forecast Horizon Scaling:")
        for result in performance_results['horizon_scaling']:
            print(f"   - {result['forecast_horizon']:2d}h horizon: {result['processing_time_seconds']:.3f}s "
                  f"({result['predictions_per_second']:.1f} pred/s)")
    
    # Display complexity analysis
    if 'computational_complexity' in performance_results:
        complexity = performance_results['computational_complexity']
        print(f"\n🧮 Computational Complexity Analysis:")
        if 'estimated_complexity' in complexity:
            print(f"   - Estimated complexity: {complexity['estimated_complexity']}")
        if 'linear_fit' in complexity and complexity['linear_fit']:
            fit = complexity['linear_fit']
            print(f"   - Linear fit R²: {fit['r_squared']:.3f}")
            print(f"   - Slope: {fit['slope']:.6f} seconds per input point")
    
    # Generate performance report
    report = analyzer.generate_performance_report(performance_results)
    print(f"\n📄 Performance Report Generated:")
    print("   - Saved detailed analysis to performance_report.md")
    
    # Save report to file
    with open("./research_validation_results/performance_report.md", "w") as f:
        f.write(report)
    
    return performance_results


async def run_cross_regional_analysis():
    """Run cross-regional optimization analysis."""
    print("\n🌍 RUNNING CROSS-REGIONAL ANALYSIS")
    print("=" * 50)
    
    # Create cross-regional optimizer
    regions = ["US-CA", "US-WA", "EU-FR", "EU-NO", "AU-NSW", "BR-RS"]
    optimizer = CrossRegionalOptimizer(regions)
    
    print(f"🗺️ Analyzing optimization across {len(regions)} regions...")
    
    # Define workload requirements
    workload_requirements = {
        "duration_hours": 72,  # 3-day training job
        "gpu_count": 16,
        "memory_gb": 128,
        "cpu_cores": 64,
        "interruptible": True
    }
    
    # Run global placement optimization
    optimization_result = await optimizer.optimize_global_placement(
        workload_requirements, forecast_horizon_hours=168  # 1 week forecast
    )
    
    print(f"✅ Cross-regional optimization completed!")
    print(f"   - Best region: {optimization_result['best_region']}")
    print(f"   - Expected carbon savings: {optimization_result['global_carbon_savings']:.1f} kg CO2")
    print(f"   - Recommended regions: {', '.join(optimization_result['recommended_regions'][:3])}")
    
    # Display regional analysis
    print(f"\n🏆 Regional Rankings:")
    regional_analysis = optimization_result['regional_analysis']
    sorted_regions = sorted(
        regional_analysis.items(), 
        key=lambda x: x[1]['optimization_score'], 
        reverse=True
    )
    
    for i, (region, analysis) in enumerate(sorted_regions[:5], 1):
        print(f"   {i}. {region}: {analysis['optimization_score']:.3f} score "
              f"({analysis['renewable_utilization']:.1%} renewable, "
              f"{analysis['recommendation']})")
    
    return optimization_result


async def main():
    """Run the complete research validation example."""
    print("🔬 CARBON-AWARE TRAINING RESEARCH VALIDATION")
    print("=" * 60)
    print("This example demonstrates comprehensive research validation")
    print("for carbon-aware machine learning training systems.\n")
    
    setup_logging()
    
    try:
        # Run all validation components
        benchmark_results = await run_benchmark_validation()
        cv_results, robustness_results = await run_statistical_validation()
        comparison_summary = await run_baseline_comparison()
        performance_results = await run_performance_analysis()
        regional_results = await run_cross_regional_analysis()
        
        print("\n🎉 VALIDATION COMPLETE")
        print("=" * 60)
        print("✅ All validation tests completed successfully!")
        print(f"✅ Results saved to: ./research_validation_results/")
        
        # Summary of key findings
        print(f"\n📋 EXECUTIVE SUMMARY:")
        print(f"   • Comprehensive benchmarks: {len(benchmark_results.get('benchmarks', {}))} test suites")
        print(f"   • Cross-validation folds: {cv_results['n_successful_folds']}/5 successful")
        print(f"   • Model robustness score: {robustness_results['robustness_score']:.3f}/1.0")
        print(f"   • Statistical comparisons: {comparison_summary['total_comparisons']} conducted")
        print(f"   • Performance scaling: {len(performance_results.get('size_scaling', []))} data points")
        print(f"   • Cross-regional analysis: {len(regional_results['regional_analysis'])} regions evaluated")
        
        print(f"\n🔬 This validation framework demonstrates:")
        print(f"   • Academic-grade statistical rigor")
        print(f"   • Comprehensive baseline comparisons") 
        print(f"   • Performance and scalability analysis")
        print(f"   • Cross-regional optimization validation")
        print(f"   • Publication-ready results and documentation")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)