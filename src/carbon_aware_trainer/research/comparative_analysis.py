"""
Comparative Analysis Framework for Carbon Forecasting Research.

This module provides tools for rigorous statistical comparison of carbon forecasting
models, including baseline comparisons, hypothesis testing, and effect size analysis
for academic publication.
"""

import asyncio
import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any, Union
from enum import Enum
import json

# Optional statistical libraries
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

from .experimental_benchmarks import ResearchMetrics, BenchmarkSuite
from ..core.types import CarbonIntensity


logger = logging.getLogger(__name__)


class StatisticalTest(Enum):
    """Types of statistical tests for model comparison."""
    PAIRED_T_TEST = "paired_t_test"
    WILCOXON_SIGNED_RANK = "wilcoxon_signed_rank"  
    DIEBOLD_MARIANO = "diebold_mariano"
    BOOTSTRAP = "bootstrap"
    PERMUTATION_TEST = "permutation_test"


@dataclass
class StatisticalResult:
    """Result of a statistical test."""
    test_type: StatisticalTest
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    interpretation: str


@dataclass
class ComparisonResult:
    """Result of comparing two forecasting models."""
    model_a: str
    model_b: str
    metric: str
    improvement_percent: float
    improvement_absolute: float
    statistical_test: StatisticalResult
    sample_size: int
    practical_significance: bool


class BaselineComparator:
    """
    Comprehensive baseline comparison framework for carbon forecasting models.
    
    Implements multiple statistical tests and effect size calculations to
    rigorously evaluate model improvements over baseline methods.
    """
    
    def __init__(self, significance_level: float = 0.05, minimum_effect_size: float = 0.1):
        """Initialize baseline comparator.
        
        Args:
            significance_level: Alpha level for statistical significance (default: 0.05)
            minimum_effect_size: Minimum effect size for practical significance
        """
        self.significance_level = significance_level
        self.minimum_effect_size = minimum_effect_size
        self.comparison_results = []
        
        logger.info(f"Initialized baseline comparator with α={significance_level}, min effect size={minimum_effect_size}")
    
    def _calculate_effect_size(self, group_a: List[float], group_b: List[float]) -> float:
        """Calculate Cohen's d effect size between two groups."""
        if not group_a or not group_b:
            return 0.0
        
        mean_a = sum(group_a) / len(group_a)
        mean_b = sum(group_b) / len(group_b)
        
        # Calculate pooled standard deviation
        var_a = sum((x - mean_a) ** 2 for x in group_a) / (len(group_a) - 1) if len(group_a) > 1 else 0
        var_b = sum((x - mean_b) ** 2 for x in group_b) / (len(group_b) - 1) if len(group_b) > 1 else 0
        
        pooled_var = ((len(group_a) - 1) * var_a + (len(group_b) - 1) * var_b) / (len(group_a) + len(group_b) - 2)
        pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else 1e-8
        
        cohen_d = (mean_a - mean_b) / pooled_std
        return cohen_d
    
    def _paired_t_test(self, differences: List[float]) -> StatisticalResult:
        """Perform paired t-test on paired differences."""
        if len(differences) < 2:
            return StatisticalResult(
                test_type=StatisticalTest.PAIRED_T_TEST,
                statistic=0.0, p_value=1.0, effect_size=0.0,
                confidence_interval=(0.0, 0.0), is_significant=False,
                interpretation="Insufficient data for t-test"
            )
        
        # Calculate t-statistic
        mean_diff = sum(differences) / len(differences)
        if abs(mean_diff) < 1e-10:
            return StatisticalResult(
                test_type=StatisticalTest.PAIRED_T_TEST,
                statistic=0.0, p_value=1.0, effect_size=0.0,
                confidence_interval=(0.0, 0.0), is_significant=False,
                interpretation="No difference between models"
            )
        
        var_diff = sum((d - mean_diff) ** 2 for d in differences) / (len(differences) - 1)
        std_error = math.sqrt(var_diff / len(differences))
        
        if std_error < 1e-10:
            t_statistic = 0.0
        else:
            t_statistic = mean_diff / std_error
        
        # Simplified p-value calculation (using normal approximation for large samples)
        if len(differences) >= 30:
            # Normal approximation
            p_value = 2 * (1 - self._normal_cdf(abs(t_statistic)))
        else:
            # Conservative p-value for small samples
            p_value = 0.1 if abs(t_statistic) > 1.0 else 0.5
        
        # 95% confidence interval
        if len(differences) >= 30:
            critical_value = 1.96  # Normal distribution
        else:
            critical_value = 2.5  # Conservative for small samples
        
        margin_error = critical_value * std_error
        ci_lower = mean_diff - margin_error
        ci_upper = mean_diff + margin_error
        
        # Effect size (standardized mean difference)
        effect_size = abs(mean_diff) / math.sqrt(var_diff) if var_diff > 0 else 0.0
        
        is_significant = p_value < self.significance_level
        
        interpretation = f"Mean difference: {mean_diff:.3f}, t({len(differences)-1}) = {t_statistic:.3f}"
        if is_significant:
            interpretation += f", p < {self.significance_level} (significant)"
        else:
            interpretation += f", p = {p_value:.3f} (not significant)"
        
        return StatisticalResult(
            test_type=StatisticalTest.PAIRED_T_TEST,
            statistic=t_statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=is_significant,
            interpretation=interpretation
        )
    
    def _normal_cdf(self, x: float) -> float:
        """Approximate normal cumulative distribution function."""
        # Using Abramowitz and Stegun approximation
        a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
        p = 0.3275911
        
        x = abs(x)
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
        
        return 0.5 + (0.5 if x >= 0 else -0.5) * y
    
    def _bootstrap_test(self, group_a: List[float], group_b: List[float], n_bootstrap: int = 1000) -> StatisticalResult:
        """Perform bootstrap hypothesis test."""
        import random
        
        observed_diff = sum(group_a) / len(group_a) - sum(group_b) / len(group_b)
        
        # Bootstrap resampling
        bootstrap_diffs = []
        combined = group_a + group_b
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            bootstrap_a = [random.choice(combined) for _ in range(len(group_a))]
            bootstrap_b = [random.choice(combined) for _ in range(len(group_b))]
            
            diff = sum(bootstrap_a) / len(bootstrap_a) - sum(bootstrap_b) / len(bootstrap_b)
            bootstrap_diffs.append(diff)
        
        # Calculate p-value (two-tailed)
        extreme_count = sum(1 for d in bootstrap_diffs if abs(d) >= abs(observed_diff))
        p_value = extreme_count / n_bootstrap
        
        # Confidence interval from bootstrap distribution
        bootstrap_diffs.sort()
        ci_lower_idx = int((self.significance_level / 2) * n_bootstrap)
        ci_upper_idx = int((1 - self.significance_level / 2) * n_bootstrap)
        ci_lower = bootstrap_diffs[ci_lower_idx] if ci_lower_idx < len(bootstrap_diffs) else bootstrap_diffs[0]
        ci_upper = bootstrap_diffs[ci_upper_idx] if ci_upper_idx < len(bootstrap_diffs) else bootstrap_diffs[-1]
        
        # Effect size
        effect_size = self._calculate_effect_size(group_a, group_b)
        
        is_significant = p_value < self.significance_level
        
        interpretation = f"Bootstrap mean difference: {observed_diff:.3f} ({n_bootstrap} resamples)"
        interpretation += f", p = {p_value:.3f}"
        
        return StatisticalResult(
            test_type=StatisticalTest.BOOTSTRAP,
            statistic=observed_diff,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=is_significant,
            interpretation=interpretation
        )
    
    async def compare_models(
        self,
        model_a_results: List[float],
        model_b_results: List[float],
        model_a_name: str,
        model_b_name: str,
        metric_name: str = "MAE",
        test_type: StatisticalTest = StatisticalTest.PAIRED_T_TEST
    ) -> ComparisonResult:
        """Compare two models using statistical testing.
        
        Args:
            model_a_results: Metric values for model A
            model_b_results: Metric values for model B
            model_a_name: Name of model A
            model_b_name: Name of model B
            metric_name: Name of the metric being compared
            test_type: Statistical test to perform
            
        Returns:
            Comparison result with statistical analysis
        """
        if len(model_a_results) != len(model_b_results):
            logger.warning(f"Unequal sample sizes: {len(model_a_results)} vs {len(model_b_results)}")
            min_len = min(len(model_a_results), len(model_b_results))
            model_a_results = model_a_results[:min_len]
            model_b_results = model_b_results[:min_len]
        
        if not model_a_results or not model_b_results:
            raise ValueError("Empty result sets provided for comparison")
        
        # Calculate improvement metrics
        mean_a = sum(model_a_results) / len(model_a_results)
        mean_b = sum(model_b_results) / len(model_b_results)
        
        improvement_absolute = mean_b - mean_a  # Positive if A is better (lower error)
        improvement_percent = (improvement_absolute / mean_b) * 100 if mean_b != 0 else 0.0
        
        # Perform statistical test
        if test_type == StatisticalTest.PAIRED_T_TEST:
            differences = [b - a for a, b in zip(model_a_results, model_b_results)]
            statistical_result = self._paired_t_test(differences)
        elif test_type == StatisticalTest.BOOTSTRAP:
            statistical_result = self._bootstrap_test(model_a_results, model_b_results)
        else:
            # Default to paired t-test
            differences = [b - a for a, b in zip(model_a_results, model_b_results)]
            statistical_result = self._paired_t_test(differences)
        
        # Determine practical significance
        practical_significance = (
            statistical_result.is_significant and 
            abs(statistical_result.effect_size) >= self.minimum_effect_size
        )
        
        comparison = ComparisonResult(
            model_a=model_a_name,
            model_b=model_b_name,
            metric=metric_name,
            improvement_percent=improvement_percent,
            improvement_absolute=improvement_absolute,
            statistical_test=statistical_result,
            sample_size=len(model_a_results),
            practical_significance=practical_significance
        )
        
        self.comparison_results.append(comparison)
        
        logger.info(f"Comparison {model_a_name} vs {model_b_name}: {improvement_percent:.1f}% improvement, p={statistical_result.p_value:.3f}")
        
        return comparison
    
    def generate_comparison_summary(self) -> Dict[str, Any]:
        """Generate summary of all model comparisons."""
        if not self.comparison_results:
            return {"message": "No comparisons performed yet"}
        
        summary = {
            "total_comparisons": len(self.comparison_results),
            "significant_improvements": 0,
            "practically_significant": 0,
            "best_improvements": [],
            "statistical_summary": {},
            "effect_sizes": []
        }
        
        for result in self.comparison_results:
            if result.statistical_test.is_significant:
                summary["significant_improvements"] += 1
            
            if result.practical_significance:
                summary["practically_significant"] += 1
            
            summary["effect_sizes"].append({
                "comparison": f"{result.model_a}_vs_{result.model_b}",
                "effect_size": result.statistical_test.effect_size,
                "improvement_percent": result.improvement_percent
            })
        
        # Find best improvements
        summary["best_improvements"] = sorted(
            [
                {
                    "comparison": f"{r.model_a} vs {r.model_b}",
                    "improvement_percent": r.improvement_percent,
                    "p_value": r.statistical_test.p_value,
                    "effect_size": r.statistical_test.effect_size
                }
                for r in self.comparison_results
            ],
            key=lambda x: x["improvement_percent"],
            reverse=True
        )[:5]
        
        return summary


class StatisticalValidator:
    """
    Statistical validation framework for carbon forecasting research.
    
    Provides comprehensive validation including cross-validation,
    significance testing, and reproducibility analysis.
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize statistical validator.
        
        Args:
            random_seed: Random seed for reproducible results
        """
        self.random_seed = random_seed
        self.validation_results = {}
        
    async def cross_validate_temporal(
        self,
        data: List[CarbonIntensity],
        model_func: callable,
        n_folds: int = 5,
        min_train_size: int = 168  # 1 week minimum training
    ) -> Dict[str, Any]:
        """Perform temporal cross-validation for time series data.
        
        Args:
            data: Time series carbon intensity data
            model_func: Function that takes training data and returns predictions
            n_folds: Number of cross-validation folds
            min_train_size: Minimum training set size (in hours)
            
        Returns:
            Cross-validation results with statistics
        """
        if len(data) < min_train_size * 2:
            raise ValueError(f"Insufficient data for cross-validation: need at least {min_train_size * 2} points")
        
        logger.info(f"Performing {n_folds}-fold temporal cross-validation on {len(data)} data points")
        
        fold_results = []
        fold_size = (len(data) - min_train_size) // n_folds
        
        for fold in range(n_folds):
            try:
                # Temporal split: train on past data, test on future
                train_end = min_train_size + fold * fold_size
                test_start = train_end
                test_end = min(test_start + fold_size, len(data))
                
                train_data = data[:train_end]
                test_data = data[test_start:test_end]
                
                if len(test_data) < 1:
                    continue
                
                # Generate predictions
                predictions = await model_func(train_data, len(test_data))
                actual_values = [ci.carbon_intensity for ci in test_data]
                
                # Calculate fold metrics
                fold_metrics = self._calculate_fold_metrics(predictions, actual_values)
                fold_metrics['fold_id'] = fold
                fold_metrics['train_size'] = len(train_data)
                fold_metrics['test_size'] = len(test_data)
                
                fold_results.append(fold_metrics)
                
            except Exception as e:
                logger.warning(f"Cross-validation fold {fold} failed: {e}")
                continue
        
        if not fold_results:
            raise ValueError("All cross-validation folds failed")
        
        # Aggregate results across folds
        cv_results = self._aggregate_cv_results(fold_results)
        cv_results['n_successful_folds'] = len(fold_results)
        cv_results['fold_details'] = fold_results
        
        logger.info(f"Cross-validation completed: {len(fold_results)}/{n_folds} successful folds")
        
        return cv_results
    
    def _calculate_fold_metrics(self, predictions: List[float], actual: List[float]) -> Dict[str, float]:
        """Calculate metrics for a single cross-validation fold."""
        if len(predictions) != len(actual) or not predictions:
            return {'mae': float('inf'), 'rmse': float('inf'), 'mape': float('inf'), 'r2': 0.0}
        
        # Basic metrics
        errors = [abs(p - a) for p, a in zip(predictions, actual)]
        mae = sum(errors) / len(errors)
        
        squared_errors = [(p - a) ** 2 for p, a in zip(predictions, actual)]
        rmse = math.sqrt(sum(squared_errors) / len(squared_errors))
        
        percentage_errors = [abs(p - a) / max(abs(a), 1e-8) for p, a in zip(predictions, actual)]
        mape = sum(percentage_errors) / len(percentage_errors)
        
        # R-squared
        actual_mean = sum(actual) / len(actual)
        ss_tot = sum((a - actual_mean) ** 2 for a in actual)
        ss_res = sum(squared_errors)
        r2 = 1.0 - (ss_res / max(ss_tot, 1e-8))
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2
        }
    
    def _aggregate_cv_results(self, fold_results: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate cross-validation results across folds."""
        metrics = ['mae', 'rmse', 'mape', 'r2']
        aggregated = {}
        
        for metric in metrics:
            values = [fold[metric] for fold in fold_results if metric in fold and not math.isinf(fold[metric])]
            
            if values:
                aggregated[f'{metric}_mean'] = sum(values) / len(values)
                aggregated[f'{metric}_std'] = math.sqrt(sum((v - aggregated[f'{metric}_mean']) ** 2 for v in values) / len(values))
                aggregated[f'{metric}_min'] = min(values)
                aggregated[f'{metric}_max'] = max(values)
            else:
                aggregated[f'{metric}_mean'] = float('inf')
                aggregated[f'{metric}_std'] = 0.0
                aggregated[f'{metric}_min'] = float('inf')
                aggregated[f'{metric}_max'] = float('inf')
        
        return aggregated
    
    async def validate_model_robustness(
        self,
        model_func: callable,
        test_data: List[CarbonIntensity],
        perturbation_levels: List[float] = None
    ) -> Dict[str, Any]:
        """Validate model robustness to input perturbations.
        
        Args:
            model_func: Model function to test
            test_data: Test data for robustness analysis
            perturbation_levels: Noise levels to test (as fraction of signal)
            
        Returns:
            Robustness analysis results
        """
        if perturbation_levels is None:
            perturbation_levels = [0.01, 0.05, 0.1, 0.2]  # 1%, 5%, 10%, 20% noise
        
        logger.info(f"Testing model robustness with {len(perturbation_levels)} perturbation levels")
        
        # Get baseline performance
        baseline_predictions = await model_func(test_data[:-24], 24)
        baseline_actual = [ci.carbon_intensity for ci in test_data[-24:]]
        baseline_metrics = self._calculate_fold_metrics(baseline_predictions, baseline_actual)
        
        robustness_results = {
            'baseline_performance': baseline_metrics,
            'perturbation_analysis': [],
            'robustness_score': 0.0
        }
        
        for noise_level in perturbation_levels:
            try:
                # Add noise to test data
                import random
                random.seed(self.random_seed)
                
                noisy_data = []
                for ci in test_data[:-24]:
                    # Add Gaussian noise
                    noise = random.gauss(0, noise_level * ci.carbon_intensity)
                    noisy_intensity = max(0, ci.carbon_intensity + noise)
                    
                    noisy_ci = CarbonIntensity(
                        carbon_intensity=noisy_intensity,
                        timestamp=ci.timestamp,
                        region=ci.region,
                        renewable_percentage=ci.renewable_percentage
                    )
                    noisy_data.append(noisy_ci)
                
                # Test model on noisy data
                noisy_predictions = await model_func(noisy_data, 24)
                noisy_metrics = self._calculate_fold_metrics(noisy_predictions, baseline_actual)
                
                # Calculate performance degradation
                mae_degradation = (noisy_metrics['mae'] - baseline_metrics['mae']) / baseline_metrics['mae']
                rmse_degradation = (noisy_metrics['rmse'] - baseline_metrics['rmse']) / baseline_metrics['rmse']
                
                perturbation_result = {
                    'noise_level': noise_level,
                    'performance_metrics': noisy_metrics,
                    'mae_degradation_percent': mae_degradation * 100,
                    'rmse_degradation_percent': rmse_degradation * 100
                }
                
                robustness_results['perturbation_analysis'].append(perturbation_result)
                
            except Exception as e:
                logger.warning(f"Robustness test failed at noise level {noise_level}: {e}")
        
        # Calculate overall robustness score (lower degradation = higher robustness)
        if robustness_results['perturbation_analysis']:
            avg_degradation = sum(
                result['mae_degradation_percent'] for result in robustness_results['perturbation_analysis']
            ) / len(robustness_results['perturbation_analysis'])
            
            # Robustness score: 1.0 = no degradation, 0.0 = complete failure
            robustness_results['robustness_score'] = max(0.0, 1.0 - avg_degradation / 100.0)
        
        return robustness_results


class PerformanceAnalyzer:
    """
    Performance analysis framework for carbon forecasting models.
    
    Analyzes computational performance, memory usage, and scalability
    characteristics for production deployment assessment.
    """
    
    def __init__(self):
        """Initialize performance analyzer."""
        self.performance_metrics = []
        
    async def profile_model_performance(
        self,
        model_func: callable,
        test_sizes: List[int] = None,
        forecast_horizons: List[int] = None
    ) -> Dict[str, Any]:
        """Profile model performance across different input sizes and horizons.
        
        Args:
            model_func: Model function to profile
            test_sizes: Input data sizes to test
            forecast_horizons: Forecast horizons to test
            
        Returns:
            Performance profiling results
        """
        if test_sizes is None:
            test_sizes = [24, 168, 720, 2160]  # 1 day, 1 week, 1 month, 3 months
        
        if forecast_horizons is None:
            forecast_horizons = [12, 24, 48, 96]  # 12h, 1d, 2d, 4d
        
        logger.info(f"Profiling model performance across {len(test_sizes)} input sizes and {len(forecast_horizons)} horizons")
        
        performance_results = {
            'size_scaling': [],
            'horizon_scaling': [],
            'memory_usage': {},
            'computational_complexity': {}
        }
        
        # Test scaling with input size
        for size in test_sizes:
            try:
                # Generate test data
                test_data = self._generate_test_data(size)
                
                # Measure performance
                start_time = datetime.now()
                predictions = await model_func(test_data, 24)  # Fixed 24h forecast
                end_time = datetime.now()
                
                processing_time = (end_time - start_time).total_seconds()
                
                performance_results['size_scaling'].append({
                    'input_size': size,
                    'processing_time_seconds': processing_time,
                    'throughput_points_per_second': size / max(processing_time, 0.001)
                })
                
            except Exception as e:
                logger.warning(f"Performance test failed for size {size}: {e}")
        
        # Test scaling with forecast horizon
        base_data = self._generate_test_data(168)  # 1 week of data
        for horizon in forecast_horizons:
            try:
                start_time = datetime.now()
                predictions = await model_func(base_data, horizon)
                end_time = datetime.now()
                
                processing_time = (end_time - start_time).total_seconds()
                
                performance_results['horizon_scaling'].append({
                    'forecast_horizon': horizon,
                    'processing_time_seconds': processing_time,
                    'predictions_per_second': horizon / max(processing_time, 0.001)
                })
                
            except Exception as e:
                logger.warning(f"Performance test failed for horizon {horizon}: {e}")
        
        # Analyze computational complexity
        performance_results['computational_complexity'] = self._analyze_complexity(
            performance_results['size_scaling']
        )
        
        return performance_results
    
    def _generate_test_data(self, size: int) -> List[CarbonIntensity]:
        """Generate synthetic test data for performance testing."""
        test_data = []
        base_time = datetime.now() - timedelta(hours=size)
        
        for i in range(size):
            # Synthetic carbon intensity with realistic patterns
            time = base_time + timedelta(hours=i)
            base_intensity = 100 + 50 * math.sin(2 * math.pi * i / 24)  # Daily pattern
            base_intensity += 20 * math.sin(2 * math.pi * i / (24 * 7))  # Weekly pattern
            base_intensity += 10 * (hash(str(i)) % 100) / 100  # Random variation
            
            ci = CarbonIntensity(
                carbon_intensity=max(10, min(400, base_intensity)),
                timestamp=time,
                region="TEST",
                renewable_percentage=0.3 + 0.4 * math.sin(2 * math.pi * i / 24)
            )
            test_data.append(ci)
        
        return test_data
    
    def _analyze_complexity(self, size_scaling_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze computational complexity from scaling results."""
        if len(size_scaling_results) < 3:
            return {"analysis": "Insufficient data for complexity analysis"}
        
        # Extract sizes and times
        sizes = [result['input_size'] for result in size_scaling_results]
        times = [result['processing_time_seconds'] for result in size_scaling_results]
        
        # Fit different complexity models (linear, quadratic, logarithmic)
        complexity_analysis = {
            'data_points': len(sizes),
            'size_range': (min(sizes), max(sizes)),
            'time_range': (min(times), max(times))
        }
        
        # Simple linear fit: time = a * size + b
        if len(sizes) >= 2:
            n = len(sizes)
            sum_size = sum(sizes)
            sum_time = sum(times)
            sum_size_time = sum(s * t for s, t in zip(sizes, times))
            sum_size_squared = sum(s * s for s in sizes)
            
            if n * sum_size_squared - sum_size * sum_size != 0:
                slope = (n * sum_size_time - sum_size * sum_time) / (n * sum_size_squared - sum_size * sum_size)
                intercept = (sum_time - slope * sum_size) / n
                
                # Calculate R-squared for linear fit
                predicted_times = [slope * s + intercept for s in sizes]
                ss_res = sum((actual - pred) ** 2 for actual, pred in zip(times, predicted_times))
                ss_tot = sum((t - sum_time / n) ** 2 for t in times)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                complexity_analysis['linear_fit'] = {
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_squared
                }
                
                # Interpret complexity
                if r_squared > 0.8:
                    if slope < 0.001:
                        complexity_analysis['estimated_complexity'] = "O(1) - Constant time"
                    elif slope < 0.01:
                        complexity_analysis['estimated_complexity'] = "O(log n) - Logarithmic time"
                    else:
                        complexity_analysis['estimated_complexity'] = "O(n) - Linear time"
                else:
                    complexity_analysis['estimated_complexity'] = "Non-linear (poor linear fit)"
        
        return complexity_analysis
    
    def generate_performance_report(self, performance_results: Dict[str, Any]) -> str:
        """Generate a readable performance analysis report."""
        report = "# Model Performance Analysis Report\n\n"
        
        # Input size scaling analysis
        if 'size_scaling' in performance_results and performance_results['size_scaling']:
            report += "## Input Size Scaling\n\n"
            report += "| Input Size | Processing Time (s) | Throughput (points/s) |\n"
            report += "|------------|--------------------|-----------------------|\n"
            
            for result in performance_results['size_scaling']:
                report += f"| {result['input_size']} | {result['processing_time_seconds']:.3f} | {result['throughput_points_per_second']:.1f} |\n"
            
            report += "\n"
        
        # Forecast horizon scaling analysis
        if 'horizon_scaling' in performance_results and performance_results['horizon_scaling']:
            report += "## Forecast Horizon Scaling\n\n"
            report += "| Horizon (hours) | Processing Time (s) | Predictions/s |\n"
            report += "|-----------------|--------------------|--------------|\n"
            
            for result in performance_results['horizon_scaling']:
                report += f"| {result['forecast_horizon']} | {result['processing_time_seconds']:.3f} | {result['predictions_per_second']:.1f} |\n"
            
            report += "\n"
        
        # Computational complexity analysis
        if 'computational_complexity' in performance_results:
            complexity = performance_results['computational_complexity']
            report += "## Computational Complexity Analysis\n\n"
            
            if 'estimated_complexity' in complexity:
                report += f"**Estimated Complexity**: {complexity['estimated_complexity']}\n\n"
            
            if 'linear_fit' in complexity:
                fit = complexity['linear_fit']
                report += f"**Linear Fit**: R² = {fit['r_squared']:.3f}\n"
                report += f"**Performance Equation**: time = {fit['slope']:.6f} × input_size + {fit['intercept']:.3f}\n\n"
        
        report += "## Recommendations\n\n"
        
        # Add performance recommendations based on results
        if 'size_scaling' in performance_results and performance_results['size_scaling']:
            max_size_result = max(performance_results['size_scaling'], key=lambda x: x['input_size'])
            if max_size_result['processing_time_seconds'] < 1.0:
                report += "- ✅ Model scales well to large input sizes\n"
            elif max_size_result['processing_time_seconds'] < 10.0:
                report += "- ⚠️ Model performance acceptable but may need optimization for very large datasets\n"
            else:
                report += "- ❌ Model may need significant optimization for production use\n"
        
        if 'computational_complexity' in performance_results:
            complexity = performance_results['computational_complexity']
            if 'estimated_complexity' in complexity:
                if 'O(1)' in complexity['estimated_complexity'] or 'O(log n)' in complexity['estimated_complexity']:
                    report += "- ✅ Excellent computational complexity for scaling\n"
                elif 'O(n)' in complexity['estimated_complexity']:
                    report += "- ✅ Good linear scaling characteristics\n"
                else:
                    report += "- ⚠️ Non-linear complexity may limit scalability\n"
        
        return report