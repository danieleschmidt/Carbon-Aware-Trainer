"""Performance optimization utilities for carbon-aware training."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import functools
import time

from .exceptions import MetricsError
from .types import CarbonIntensity, OptimalWindow


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization."""
    operation: str
    duration_ms: float
    memory_mb: float
    cpu_percent: float
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PerformanceProfiler:
    """Performance profiling and optimization guidance."""
    
    def __init__(self):
        """Initialize performance profiler."""
        self.metrics: List[PerformanceMetrics] = []
        self._operation_stats: Dict[str, List[float]] = {}
    
    async def profile_async(self, operation: str, func: Callable, *args, **kwargs) -> Tuple[Any, PerformanceMetrics]:
        """Profile async function execution.
        
        Args:
            operation: Operation name for tracking
            func: Async function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (result, metrics)
        """
        import psutil
        
        # Pre-execution metrics
        process = psutil.Process()
        start_memory = process.memory_info().rss / (1024 * 1024)  # MB
        start_time = time.perf_counter()
        start_cpu = process.cpu_percent()
        
        result = None
        error = None
        success = True
        
        try:
            result = await func(*args, **kwargs)
        except Exception as e:
            error = str(e)
            success = False
            raise
        finally:
            # Post-execution metrics
            end_time = time.perf_counter()
            end_memory = process.memory_info().rss / (1024 * 1024)  # MB
            end_cpu = process.cpu_percent()
            
            duration_ms = (end_time - start_time) * 1000
            memory_delta = end_memory - start_memory
            cpu_avg = (start_cpu + end_cpu) / 2
            
            metrics = PerformanceMetrics(
                operation=operation,
                duration_ms=duration_ms,
                memory_mb=memory_delta,
                cpu_percent=cpu_avg,
                success=success,
                error=error
            )
            
            self.metrics.append(metrics)
            
            # Update operation statistics
            if operation not in self._operation_stats:
                self._operation_stats[operation] = []
            self._operation_stats[operation].append(duration_ms)
            
            logger.debug(f"Profiled {operation}: {duration_ms:.2f}ms, {memory_delta:.2f}MB")
        
        return result, metrics
    
    def get_operation_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for a specific operation.
        
        Args:
            operation: Operation name
            
        Returns:
            Dictionary with operation statistics
        """
        if operation not in self._operation_stats:
            return {"error": f"No data for operation: {operation}"}
        
        durations = self._operation_stats[operation]
        
        return {
            "count": len(durations),
            "avg_duration_ms": sum(durations) / len(durations),
            "min_duration_ms": min(durations),
            "max_duration_ms": max(durations),
            "total_duration_ms": sum(durations)
        }
    
    def get_slowest_operations(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Get slowest operations by average duration.
        
        Args:
            limit: Maximum number of operations to return
            
        Returns:
            List of (operation, avg_duration_ms) tuples
        """
        operation_avgs = []
        
        for operation, durations in self._operation_stats.items():
            avg_duration = sum(durations) / len(durations)
            operation_avgs.append((operation, avg_duration))
        
        # Sort by duration (descending)
        operation_avgs.sort(key=lambda x: x[1], reverse=True)
        
        return operation_avgs[:limit]


class AsyncBatchProcessor:
    """Optimized batch processing for async operations."""
    
    def __init__(
        self,
        batch_size: int = 10,
        max_concurrent: int = 5,
        timeout: Optional[float] = None
    ):
        """Initialize batch processor.
        
        Args:
            batch_size: Items per batch
            max_concurrent: Maximum concurrent batches
            timeout: Timeout per batch in seconds
        """
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_items(
        self,
        items: List[Any],
        process_func: Callable,
        *args,
        **kwargs
    ) -> List[Any]:
        """Process items in optimized batches.
        
        Args:
            items: Items to process
            process_func: Async function to process each item
            *args: Additional arguments for process_func
            **kwargs: Additional keyword arguments for process_func
            
        Returns:
            List of processed results
        """
        # Split items into batches
        batches = [
            items[i:i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]
        
        # Process batches concurrently
        tasks = [
            self._process_batch(batch, process_func, *args, **kwargs)
            for batch in batches
        ]
        
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results
        results = []
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                logger.error(f"Batch processing error: {batch_result}")
                continue
            results.extend(batch_result)
        
        return results
    
    async def _process_batch(
        self,
        batch: List[Any],
        process_func: Callable,
        *args,
        **kwargs
    ) -> List[Any]:
        """Process a single batch with concurrency control.
        
        Args:
            batch: Batch of items to process
            process_func: Processing function
            *args: Additional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            List of batch results
        """
        async with self._semaphore:
            tasks = [
                self._process_item_with_timeout(process_func, item, *args, **kwargs)
                for item in batch
            ]
            
            return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _process_item_with_timeout(
        self,
        process_func: Callable,
        item: Any,
        *args,
        **kwargs
    ) -> Any:
        """Process single item with timeout.
        
        Args:
            process_func: Processing function
            item: Item to process
            *args: Additional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Processed result
        """
        try:
            if self.timeout:
                return await asyncio.wait_for(
                    process_func(item, *args, **kwargs),
                    timeout=self.timeout
                )
            else:
                return await process_func(item, *args, **kwargs)
        except asyncio.TimeoutError:
            logger.warning(f"Item processing timeout: {self.timeout}s")
            raise
        except Exception as e:
            logger.error(f"Item processing error: {e}")
            raise


class ParallelCarbon Forecaster:
    """Parallel carbon intensity forecasting for multiple regions."""
    
    def __init__(self, max_workers: int = None):
        """Initialize parallel forecaster.
        
        Args:
            max_workers: Maximum parallel workers (defaults to CPU count)
        """
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self._thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self._process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
    
    async def forecast_multiple_regions(
        self,
        regions: List[str],
        forecast_func: Callable,
        duration: timedelta = timedelta(hours=24),
        use_processes: bool = False
    ) -> Dict[str, Any]:
        """Forecast carbon intensity for multiple regions in parallel.
        
        Args:
            regions: List of region codes
            forecast_func: Function to get forecast for single region
            duration: Forecast duration
            use_processes: Whether to use process pool (for CPU-intensive work)
            
        Returns:
            Dictionary mapping regions to forecast results
        """
        loop = asyncio.get_event_loop()
        executor = self._process_pool if use_processes else self._thread_pool
        
        # Create tasks for each region
        tasks = []
        for region in regions:
            task = loop.run_in_executor(
                executor,
                functools.partial(forecast_func, region, duration)
            )
            tasks.append((region, task))
        
        # Execute in parallel
        results = {}
        for region, task in tasks:
            try:
                result = await task
                results[region] = result
            except Exception as e:
                logger.error(f"Forecast error for {region}: {e}")
                results[region] = {"error": str(e)}
        
        return results
    
    async def find_optimal_windows_parallel(
        self,
        regions: List[str],
        window_finder_func: Callable,
        duration_hours: int = 8,
        **kwargs
    ) -> Dict[str, List[OptimalWindow]]:
        """Find optimal training windows for multiple regions in parallel.
        
        Args:
            regions: List of region codes
            window_finder_func: Function to find windows for single region
            duration_hours: Training duration in hours
            **kwargs: Additional arguments for window finder
            
        Returns:
            Dictionary mapping regions to optimal windows
        """
        loop = asyncio.get_event_loop()
        
        # Create tasks for each region
        tasks = []
        for region in regions:
            task = loop.run_in_executor(
                self._thread_pool,
                functools.partial(
                    window_finder_func,
                    region=region,
                    duration_hours=duration_hours,
                    **kwargs
                )
            )
            tasks.append((region, task))
        
        # Execute in parallel
        results = {}
        for region, task in tasks:
            try:
                windows = await task
                results[region] = windows
            except Exception as e:
                logger.error(f"Window finding error for {region}: {e}")
                results[region] = []
        
        return results
    
    def cleanup(self):
        """Clean up executor resources."""
        self._thread_pool.shutdown(wait=True)
        self._process_pool.shutdown(wait=True)


class AdaptiveBatchSizer:
    """Dynamically adjust batch sizes based on performance and carbon intensity."""
    
    def __init__(
        self,
        initial_batch_size: int = 32,
        min_batch_size: int = 8,
        max_batch_size: int = 256,
        performance_target_ms: float = 1000.0,
        carbon_sensitivity: float = 0.2
    ):
        """Initialize adaptive batch sizer.
        
        Args:
            initial_batch_size: Starting batch size
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
            performance_target_ms: Target step time in milliseconds
            carbon_sensitivity: Sensitivity to carbon intensity changes (0-1)
        """
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.performance_target_ms = performance_target_ms
        self.carbon_sensitivity = carbon_sensitivity
        
        # Performance tracking
        self._recent_step_times: List[float] = []
        self._recent_carbon_intensities: List[float] = []
        self._adjustment_history: List[Tuple[datetime, int, str]] = []
    
    def update_metrics(
        self,
        step_time_ms: float,
        carbon_intensity: Optional[float] = None
    ) -> None:
        """Update performance metrics.
        
        Args:
            step_time_ms: Training step time in milliseconds
            carbon_intensity: Current carbon intensity
        """
        self._recent_step_times.append(step_time_ms)
        if len(self._recent_step_times) > 20:  # Keep last 20 measurements
            self._recent_step_times.pop(0)
        
        if carbon_intensity is not None:
            self._recent_carbon_intensities.append(carbon_intensity)
            if len(self._recent_carbon_intensities) > 20:
                self._recent_carbon_intensities.pop(0)
    
    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size based on current conditions.
        
        Returns:
            Recommended batch size
        """
        if len(self._recent_step_times) < 3:
            return self.current_batch_size
        
        # Calculate average step time
        avg_step_time = sum(self._recent_step_times) / len(self._recent_step_times)
        
        # Performance adjustment
        performance_adjustment = 0
        if avg_step_time > self.performance_target_ms * 1.2:
            # Too slow, reduce batch size
            performance_adjustment = -1
        elif avg_step_time < self.performance_target_ms * 0.8:
            # Too fast, increase batch size
            performance_adjustment = 1
        
        # Carbon intensity adjustment
        carbon_adjustment = 0
        if len(self._recent_carbon_intensities) >= 2:
            recent_carbon = self._recent_carbon_intensities[-1]
            prev_carbon = self._recent_carbon_intensities[-2]
            
            carbon_change = (recent_carbon - prev_carbon) / prev_carbon
            
            if carbon_change > 0.1:  # Carbon intensity increased significantly
                # Reduce batch size to finish training faster
                carbon_adjustment = -int(self.carbon_sensitivity * 2)
            elif carbon_change < -0.1:  # Carbon intensity decreased significantly
                # Increase batch size to be more efficient
                carbon_adjustment = int(self.carbon_sensitivity * 2)
        
        # Apply adjustments
        total_adjustment = performance_adjustment + carbon_adjustment
        
        if total_adjustment != 0:
            new_batch_size = self.current_batch_size + (total_adjustment * 4)  # Adjust by steps of 4
            new_batch_size = max(self.min_batch_size, min(new_batch_size, self.max_batch_size))
            
            if new_batch_size != self.current_batch_size:
                reason = []
                if performance_adjustment != 0:
                    reason.append(f"performance ({avg_step_time:.1f}ms)")
                if carbon_adjustment != 0:
                    reason.append(f"carbon ({self._recent_carbon_intensities[-1]:.1f})")
                
                self._adjustment_history.append((
                    datetime.now(),
                    new_batch_size,
                    ", ".join(reason)
                ))
                
                logger.info(
                    f"Adjusted batch size: {self.current_batch_size} -> {new_batch_size} "
                    f"(reason: {', '.join(reason)})"
                )
                
                self.current_batch_size = new_batch_size
        
        return self.current_batch_size
    
    def get_adjustment_history(self) -> List[Dict[str, Any]]:
        """Get batch size adjustment history.
        
        Returns:
            List of adjustment records
        """
        return [
            {
                "timestamp": timestamp.isoformat(),
                "batch_size": batch_size,
                "reason": reason
            }
            for timestamp, batch_size, reason in self._adjustment_history
        ]


class ConcurrentCarbonMonitor:
    """Concurrent carbon intensity monitoring for multiple regions."""
    
    def __init__(self, max_concurrent_requests: int = 10):
        """Initialize concurrent monitor.
        
        Args:
            max_concurrent_requests: Maximum concurrent API requests
        """
        self.max_concurrent_requests = max_concurrent_requests
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._request_times: List[float] = []
    
    async def monitor_regions_concurrent(
        self,
        regions: List[str],
        monitor_func: Callable,
        **kwargs
    ) -> Dict[str, CarbonIntensity]:
        """Monitor carbon intensity for multiple regions concurrently.
        
        Args:
            regions: List of region codes
            monitor_func: Function to get carbon intensity for single region
            **kwargs: Additional arguments for monitor function
            
        Returns:
            Dictionary mapping regions to carbon intensity
        """
        # Create monitoring tasks
        tasks = [
            self._monitor_region_with_semaphore(region, monitor_func, **kwargs)
            for region in regions
        ]
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        region_intensities = {}
        for region, result in zip(regions, results):
            if isinstance(result, Exception):
                logger.error(f"Monitoring error for {region}: {result}")
                continue
            
            region_intensities[region] = result
        
        return region_intensities
    
    async def _monitor_region_with_semaphore(
        self,
        region: str,
        monitor_func: Callable,
        **kwargs
    ) -> CarbonIntensity:
        """Monitor single region with concurrency control.
        
        Args:
            region: Region code
            monitor_func: Monitoring function
            **kwargs: Additional arguments
            
        Returns:
            Carbon intensity for region
        """
        async with self._semaphore:
            start_time = time.perf_counter()
            
            try:
                result = await monitor_func(region, **kwargs)
                return result
            finally:
                request_time = time.perf_counter() - start_time
                self._request_times.append(request_time)
                
                # Keep only recent request times
                if len(self._request_times) > 100:
                    self._request_times.pop(0)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get monitoring performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self._request_times:
            return {"error": "No request data available"}
        
        return {
            "avg_request_time_ms": (sum(self._request_times) / len(self._request_times)) * 1000,
            "min_request_time_ms": min(self._request_times) * 1000,
            "max_request_time_ms": max(self._request_times) * 1000,
            "total_requests": len(self._request_times),
            "requests_per_second": len(self._request_times) / sum(self._request_times) if self._request_times else 0
        }