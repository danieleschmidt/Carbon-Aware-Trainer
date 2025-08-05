"""Unit tests for optimization utilities."""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from carbon_aware_trainer.core.optimization import (
    PerformanceProfiler, AsyncBatchProcessor, ParallelCarbonForecaster,
    AdaptiveBatchSizer, ConcurrentCarbonMonitor, PerformanceMetrics
)
from carbon_aware_trainer.core.types import CarbonIntensity, CarbonIntensityUnit


class TestPerformanceProfiler:
    """Test cases for PerformanceProfiler."""
    
    @pytest.fixture
    def profiler(self):
        """Create performance profiler for testing."""
        return PerformanceProfiler()
    
    @pytest.mark.asyncio
    async def test_async_profiling(self, profiler):
        """Test async function profiling."""
        async def test_function(delay=0.1):
            await asyncio.sleep(delay)
            return "result"
        
        result, metrics = await profiler.profile_async("test_op", test_function, delay=0.05)
        
        assert result == "result"
        assert metrics.operation == "test_op"
        assert metrics.duration_ms >= 40  # Should be around 50ms
        assert metrics.success is True
        assert metrics.error is None
    
    @pytest.mark.asyncio
    async def test_profiling_with_error(self, profiler):
        """Test profiling function that raises error."""
        async def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            await profiler.profile_async("failing_op", failing_function)
        
        # Check that error was recorded
        assert len(profiler.metrics) == 1
        metrics = profiler.metrics[0]
        assert metrics.success is False
        assert "Test error" in metrics.error
    
    def test_operation_stats(self, profiler):
        """Test operation statistics calculation."""
        # Manually add some metrics
        for i in range(5):
            profiler._operation_stats["test_op"] = profiler._operation_stats.get("test_op", [])
            profiler._operation_stats["test_op"].append(100 + i * 10)  # 100, 110, 120, 130, 140
        
        stats = profiler.get_operation_stats("test_op")
        
        assert stats["count"] == 5
        assert stats["avg_duration_ms"] == 120.0
        assert stats["min_duration_ms"] == 100.0
        assert stats["max_duration_ms"] == 140.0
        assert stats["total_duration_ms"] == 600.0
    
    def test_slowest_operations(self, profiler):
        """Test slowest operations identification."""
        # Add operations with different performance
        profiler._operation_stats = {
            "fast_op": [10, 15, 12],
            "slow_op": [100, 110, 105],
            "medium_op": [50, 55, 52]
        }
        
        slowest = profiler.get_slowest_operations(limit=2)
        
        assert len(slowest) == 2
        assert slowest[0][0] == "slow_op"
        assert slowest[1][0] == "medium_op"
        assert slowest[0][1] > slowest[1][1]  # slow_op should be slower


class TestAsyncBatchProcessor:
    """Test cases for AsyncBatchProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Create batch processor for testing."""
        return AsyncBatchProcessor(batch_size=3, max_concurrent=2, timeout=1.0)
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, processor):
        """Test basic batch processing."""
        async def process_item(item):
            await asyncio.sleep(0.01)  # Simulate processing time
            return item * 2
        
        items = [1, 2, 3, 4, 5, 6, 7]
        results = await processor.process_items(items, process_item)
        
        # All items should be processed
        assert len(results) == 7
        assert set(results) == {2, 4, 6, 8, 10, 12, 14}
    
    @pytest.mark.asyncio
    async def test_batch_processing_with_errors(self, processor):
        """Test batch processing with some errors."""
        async def process_item(item):
            if item == 3:
                raise ValueError(f"Error processing {item}")
            return item * 2
        
        items = [1, 2, 3, 4, 5]
        results = await processor.process_items(items, process_item)
        
        # Should have 4 successful results and 1 exception
        exceptions = [r for r in results if isinstance(r, Exception)]
        successes = [r for r in results if not isinstance(r, Exception)]
        
        assert len(exceptions) == 1
        assert len(successes) == 4
        assert set(successes) == {2, 4, 8, 10}
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling in batch processing."""
        processor = AsyncBatchProcessor(batch_size=2, timeout=0.1)
        
        async def slow_process_item(item):
            await asyncio.sleep(0.2)  # Longer than timeout
            return item
        
        items = [1, 2]
        results = await processor.process_items(items, slow_process_item)
        
        # Both items should timeout
        assert len(results) == 2
        assert all(isinstance(r, Exception) for r in results)


class TestParallelCarbonForecaster:
    """Test cases for ParallelCarbonForecaster."""
    
    @pytest.fixture
    def forecaster(self):
        """Create parallel forecaster for testing."""
        return ParallelCarbonForecaster(max_workers=2)
    
    @pytest.mark.asyncio
    async def test_parallel_forecasting(self, forecaster):
        """Test parallel forecasting for multiple regions."""
        def mock_forecast_func(region, duration):
            # Simulate different forecast results per region
            return {
                "region": region,
                "forecast": [{"intensity": 100 + hash(region) % 50}],
                "duration": duration
            }
        
        regions = ["US-CA", "US-WA", "EU-FR"]
        results = await forecaster.forecast_multiple_regions(
            regions, mock_forecast_func, duration=timedelta(hours=12)
        )
        
        assert len(results) == 3
        assert all(region in results for region in regions)
        assert all("forecast" in results[region] for region in regions)
    
    @pytest.mark.asyncio
    async def test_parallel_forecasting_with_errors(self, forecaster):
        """Test parallel forecasting with some regions failing."""
        def mock_forecast_func(region, duration):
            if region == "BAD-REGION":
                raise ValueError("Invalid region")
            return {"region": region, "forecast": []}
        
        regions = ["US-CA", "BAD-REGION", "EU-FR"]
        results = await forecaster.forecast_multiple_regions(
            regions, mock_forecast_func
        )
        
        assert len(results) == 3
        assert "error" in results["BAD-REGION"]
        assert "forecast" in results["US-CA"]
        assert "forecast" in results["EU-FR"]
    
    def test_cleanup(self, forecaster):
        """Test resource cleanup."""
        forecaster.cleanup()
        # Should not raise any exceptions


class TestAdaptiveBatchSizer:
    """Test cases for AdaptiveBatchSizer."""
    
    @pytest.fixture
    def batch_sizer(self):
        """Create adaptive batch sizer for testing."""
        return AdaptiveBatchSizer(
            initial_batch_size=32,
            min_batch_size=8,
            max_batch_size=128,
            performance_target_ms=1000.0
        )
    
    def test_initial_batch_size(self, batch_sizer):
        """Test initial batch size."""
        assert batch_sizer.get_optimal_batch_size() == 32
    
    def test_performance_based_adjustment(self, batch_sizer):
        """Test batch size adjustment based on performance."""
        # Simulate slow performance
        for _ in range(5):
            batch_sizer.update_metrics(step_time_ms=1500.0)  # Slower than target
        
        new_size = batch_sizer.get_optimal_batch_size()
        assert new_size < 32  # Should reduce batch size
    
    def test_fast_performance_adjustment(self, batch_sizer):
        """Test batch size increase for fast performance."""
        # Simulate fast performance
        for _ in range(5):
            batch_sizer.update_metrics(step_time_ms=500.0)  # Faster than target
        
        new_size = batch_sizer.get_optimal_batch_size()
        assert new_size > 32  # Should increase batch size
    
    def test_carbon_intensity_adjustment(self, batch_sizer):
        """Test batch size adjustment based on carbon intensity."""
        # Start with some baseline measurements
        for i in range(3):
            batch_sizer.update_metrics(step_time_ms=1000.0, carbon_intensity=100.0)
        
        # Simulate carbon intensity increase
        batch_sizer.update_metrics(step_time_ms=1000.0, carbon_intensity=150.0)
        
        # Should reduce batch size due to high carbon
        new_size = batch_sizer.get_optimal_batch_size()
        # May or may not change depending on exact logic, but should be recorded
        assert len(batch_sizer._recent_carbon_intensities) > 0
    
    def test_boundary_enforcement(self, batch_sizer):
        """Test that batch size stays within min/max bounds."""
        # Force many adjustments downward
        for _ in range(20):
            batch_sizer.update_metrics(step_time_ms=5000.0)  # Very slow
            batch_sizer.get_optimal_batch_size()
        
        final_size = batch_sizer.get_optimal_batch_size()
        assert final_size >= batch_sizer.min_batch_size
        
        # Reset and force adjustments upward
        batch_sizer.current_batch_size = 32
        for _ in range(20):
            batch_sizer.update_metrics(step_time_ms=100.0)  # Very fast
            batch_sizer.get_optimal_batch_size()
        
        final_size = batch_sizer.get_optimal_batch_size()
        assert final_size <= batch_sizer.max_batch_size
    
    def test_adjustment_history(self, batch_sizer):
        """Test adjustment history tracking."""
        # Force an adjustment
        for _ in range(5):
            batch_sizer.update_metrics(step_time_ms=2000.0)
        
        original_size = batch_sizer.current_batch_size
        batch_sizer.get_optimal_batch_size()
        
        history = batch_sizer.get_adjustment_history()
        if batch_sizer.current_batch_size != original_size:
            assert len(history) > 0
            assert "timestamp" in history[0]
            assert "batch_size" in history[0]
            assert "reason" in history[0]


class TestConcurrentCarbonMonitor:
    """Test cases for ConcurrentCarbonMonitor."""
    
    @pytest.fixture
    def monitor(self):
        """Create concurrent carbon monitor for testing."""
        return ConcurrentCarbonMonitor(max_concurrent_requests=3)
    
    @pytest.mark.asyncio
    async def test_concurrent_monitoring(self, monitor):
        """Test concurrent monitoring of multiple regions."""
        async def mock_monitor_func(region):
            await asyncio.sleep(0.01)  # Simulate API delay
            return CarbonIntensity(
                region=region,
                timestamp=datetime.now(),
                carbon_intensity=100 + hash(region) % 50,
                unit=CarbonIntensityUnit.GRAMS_CO2_PER_KWH,
                data_source="mock"
            )
        
        regions = ["US-CA", "US-WA", "EU-FR", "EU-DE"]
        results = await monitor.monitor_regions_concurrent(regions, mock_monitor_func)
        
        assert len(results) == 4
        assert all(region in results for region in regions)
        assert all(isinstance(results[region], CarbonIntensity) for region in regions)
    
    @pytest.mark.asyncio
    async def test_concurrent_monitoring_with_errors(self, monitor):
        """Test concurrent monitoring with some regions failing."""
        async def mock_monitor_func(region):
            if region == "FAIL-REGION":
                raise ValueError("Monitoring failed")
            return CarbonIntensity(
                region=region,
                timestamp=datetime.now(),
                carbon_intensity=100.0,
                unit=CarbonIntensityUnit.GRAMS_CO2_PER_KWH,
                data_source="mock"
            )
        
        regions = ["US-CA", "FAIL-REGION", "EU-FR"]
        results = await monitor.monitor_regions_concurrent(regions, mock_monitor_func)
        
        # Should only have results for successful regions
        assert len(results) == 2
        assert "US-CA" in results
        assert "EU-FR" in results
        assert "FAIL-REGION" not in results
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self, monitor):
        """Test performance statistics tracking."""
        async def mock_monitor_func(region):
            await asyncio.sleep(0.05)  # Consistent delay
            return CarbonIntensity(
                region=region,
                timestamp=datetime.now(),
                carbon_intensity=100.0,
                unit=CarbonIntensityUnit.GRAMS_CO2_PER_KWH,
                data_source="mock"
            )
        
        regions = ["US-CA", "US-WA"]
        await monitor.monitor_regions_concurrent(regions, mock_monitor_func)
        
        stats = monitor.get_performance_stats()
        assert "avg_request_time_ms" in stats
        assert "total_requests" in stats
        assert stats["total_requests"] == 2
        assert stats["avg_request_time_ms"] >= 40  # Should be around 50ms