"""Performance and load testing for carbon-aware trainer."""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import tempfile
from pathlib import Path

from carbon_aware_trainer.core.cache import MemoryCache, MultiLevelCache, PersistentCache
from carbon_aware_trainer.core.optimization import (
    AsyncBatchProcessor, ConcurrentCarbonMonitor, AdaptiveBatchSizer
)
from carbon_aware_trainer.core.monitor import CarbonMonitor
from carbon_aware_trainer.core.types import CarbonDataSource, CarbonIntensity, CarbonIntensityUnit
from carbon_aware_trainer.carbon_models.cached import CachedProvider


class TestPerformanceScaling:
    """Test performance under load and scaling scenarios."""
    
    @pytest.mark.asyncio
    async def test_cache_performance_large_dataset(self):
        """Test cache performance with large datasets."""
        cache = MemoryCache(max_size=10000, max_memory_mb=50)
        
        # Measure performance of large number of operations
        start_time = time.perf_counter()
        
        # Write performance test
        write_tasks = []
        for i in range(1000):
            task = cache.set(f"key_{i}", f"value_{i}_{'x' * 100}")  # ~100 char values
            write_tasks.append(task)
        
        await asyncio.gather(*write_tasks)
        
        write_time = time.perf_counter() - start_time
        
        # Read performance test
        start_time = time.perf_counter()
        
        read_tasks = []
        for i in range(1000):
            task = cache.get(f"key_{i}")
            read_tasks.append(task)
        
        results = await asyncio.gather(*read_tasks)
        
        read_time = time.perf_counter() - start_time
        
        # Performance assertions
        assert write_time < 2.0  # Should complete within 2 seconds
        assert read_time < 1.0   # Reads should be faster
        assert len([r for r in results if r is not None]) >= 900  # Most should be cached
        
        # Check cache stats
        stats = cache.get_stats()
        assert stats['hit_rate'] > 0.9  # High hit rate expected
    
    @pytest.mark.asyncio
    async def test_concurrent_cache_access(self):
        """Test cache performance under concurrent access."""
        cache = MemoryCache(max_size=1000)
        
        async def worker(worker_id, operations=100):
            """Worker function for concurrent testing."""
            for i in range(operations):
                key = f"worker_{worker_id}_key_{i}"
                value = f"worker_{worker_id}_value_{i}"
                
                # Mix of read and write operations
                if i % 3 == 0:
                    await cache.get(key)  # Read (may miss)
                else:
                    await cache.set(key, value)  # Write
        
        # Run multiple workers concurrently
        start_time = time.perf_counter()
        
        tasks = [worker(i, 200) for i in range(10)]  # 10 workers, 200 ops each
        await asyncio.gather(*tasks)
        
        total_time = time.perf_counter() - start_time
        
        # Should handle 2000 operations reasonably quickly
        assert total_time < 5.0
        
        stats = cache.get_stats()
        assert stats['entries'] <= 1000  # Respects max size
    
    @pytest.mark.asyncio
    async def test_batch_processor_scalability(self):
        """Test batch processor with large workloads."""
        processor = AsyncBatchProcessor(batch_size=50, max_concurrent=10)
        
        async def process_item(item):
            # Simulate varying processing times
            await asyncio.sleep(0.001 + (item % 10) * 0.0001)
            return item ** 2
        
        # Process large number of items
        items = list(range(2000))
        
        start_time = time.perf_counter()
        results = await processor.process_items(items, process_item)
        total_time = time.perf_counter() - start_time
        
        # Verify results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == 2000
        
        # Should be faster than sequential processing
        # Sequential would take at least 2000 * 0.001 = 2 seconds
        assert total_time < 1.5  # Parallel should be significantly faster
    
    @pytest.mark.asyncio
    async def test_concurrent_carbon_monitoring_load(self):
        """Test concurrent carbon monitoring under load."""
        monitor = ConcurrentCarbonMonitor(max_concurrent_requests=20)
        
        async def mock_monitor_func(region):
            # Simulate API latency
            await asyncio.sleep(0.01 + (hash(region) % 10) * 0.001)
            return CarbonIntensity(
                region=region,
                timestamp=datetime.now(),
                carbon_intensity=100 + hash(region) % 100,
                unit=CarbonIntensityUnit.GRAMS_CO2_PER_KWH,
                data_source="mock"
            )
        
        # Test with many regions
        regions = [f"REGION_{i}" for i in range(100)]
        
        start_time = time.perf_counter()
        results = await monitor.monitor_regions_concurrent(regions, mock_monitor_func)
        total_time = time.perf_counter() - start_time
        
        # Should handle 100 regions efficiently
        assert len(results) == 100
        assert total_time < 2.0  # Should complete within 2 seconds
        
        # Check performance stats
        stats = monitor.get_performance_stats()
        assert stats['total_requests'] == 100
        assert stats['requests_per_second'] > 50  # Good throughput
    
    @pytest.mark.asyncio
    async def test_adaptive_batch_sizer_responsiveness(self):
        """Test adaptive batch sizer responsiveness to changing conditions."""
        sizer = AdaptiveBatchSizer(
            initial_batch_size=32,
            min_batch_size=8,
            max_batch_size=256,
            performance_target_ms=1000.0
        )
        
        # Simulate rapidly changing conditions
        conditions = [
            (2000.0, 150.0),  # Slow performance, high carbon
            (500.0, 50.0),    # Fast performance, low carbon
            (1500.0, 200.0),  # Slow performance, very high carbon
            (800.0, 80.0),    # Good performance, moderate carbon
        ] * 10  # Repeat pattern
        
        batch_sizes = []
        
        for step_time, carbon_intensity in conditions:
            sizer.update_metrics(step_time, carbon_intensity)
            batch_size = sizer.get_optimal_batch_size()
            batch_sizes.append(batch_size)
        
        # Should show adaptation over time
        assert len(set(batch_sizes)) > 1  # Batch size should change
        assert min(batch_sizes) >= sizer.min_batch_size
        assert max(batch_sizes) <= sizer.max_batch_size
        
        # Should have adjustment history
        history = sizer.get_adjustment_history()
        assert len(history) > 0
    
    @pytest.mark.asyncio
    async def test_persistent_cache_large_dataset(self):
        """Test persistent cache with large dataset."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "large_test.db"
            cache = PersistentCache(db_path, max_size=5000)
            
            # Write large amount of data
            start_time = time.perf_counter()
            
            for i in range(1000):
                large_value = {"data": "x" * 1000, "id": i, "metadata": {"created": datetime.now().isoformat()}}
                await cache.set(f"large_key_{i}", large_value)
            
            write_time = time.perf_counter() - start_time
            
            # Read back data
            start_time = time.perf_counter()
            
            results = []
            for i in range(0, 1000, 10):  # Sample every 10th item
                result = await cache.get(f"large_key_{i}")
                results.append(result)
            
            read_time = time.perf_counter() - start_time
            
            # Performance and correctness checks
            assert write_time < 10.0  # Should complete within 10 seconds
            assert read_time < 2.0    # Reads should be faster
            assert len([r for r in results if r is not None]) == 100
            
            # Check database stats
            stats = cache.get_stats()
            assert stats['entries'] == 1000
    
    @pytest.mark.asyncio
    async def test_multi_level_cache_performance(self):
        """Test multi-level cache performance characteristics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multi-level cache
            memory_cache = MemoryCache(max_size=100)  # Small L1
            persistent_cache = PersistentCache(Path(temp_dir) / "l2.db", max_size=1000)
            cache = MultiLevelCache(memory_cache, persistent_cache)
            
            # Fill cache with data
            for i in range(200):
                await cache.set(f"key_{i}", f"value_{i}")
            
            # Test performance patterns
            start_time = time.perf_counter()
            
            # Access recently set items (should be in L1)
            l1_results = []
            for i in range(190, 200):
                result = await cache.get(f"key_{i}")
                l1_results.append(result)
            
            l1_time = time.perf_counter() - start_time
            
            # Access older items (should be in L2)
            start_time = time.perf_counter()
            
            l2_results = []
            for i in range(0, 10):
                result = await cache.get(f"key_{i}")
                l2_results.append(result)
            
            l2_time = time.perf_counter() - start_time
            
            # L1 hits should be faster than L2 hits
            assert all(r is not None for r in l1_results)
            assert all(r is not None for r in l2_results)
            
            # Get cache statistics
            stats = cache.get_stats()
            assert stats['l1_hits'] > 0
            assert stats['l2_hits'] > 0
            assert stats['hit_rate'] > 0.95  # Very high hit rate expected
    
    @pytest.mark.asyncio
    async def test_carbon_monitoring_stress_test(self):
        """Stress test carbon monitoring with rapid updates."""
        # Create sample data for stress testing
        sample_data = {
            "regions": {
                f"REGION_{i}": {
                    "historical": [
                        {
                            "timestamp": (datetime.now() - timedelta(hours=j)).isoformat(),
                            "carbon_intensity": 50 + (i * 10) + (j * 5),
                            "renewable_percentage": 40 + (i % 40),
                            "confidence": 0.8 + (i % 20) * 0.01
                        }
                        for j in range(24)  # 24 hours of data
                    ],
                    "forecast": []
                }
                for i in range(20)  # 20 regions
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump(sample_data, f)
            temp_file = f.name
        
        try:
            monitor = CarbonMonitor(
                regions=[f"REGION_{i}" for i in range(20)],
                data_source=CarbonDataSource.CACHED,
                api_key=temp_file,
                update_interval=1  # Very frequent updates
            )
            
            async with monitor:
                await monitor.start_monitoring()
                
                # Let it run for a short time with rapid updates
                await asyncio.sleep(3)
                
                # Rapidly query all regions multiple times
                start_time = time.perf_counter()
                
                for _ in range(50):  # 50 rapid queries
                    tasks = [
                        monitor.get_current_intensity(f"REGION_{i}")
                        for i in range(20)
                    ]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Should get mostly successful results
                    successful = [r for r in results if not isinstance(r, Exception)]
                    assert len(successful) >= 18  # At least 90% success rate
                
                query_time = time.perf_counter() - start_time
                
                await monitor.stop_monitoring()
                
                # Should handle rapid queries efficiently
                assert query_time < 5.0  # 1000 queries in under 5 seconds
                
                # Check final status
                status = monitor.get_current_status()
                assert len(status['current_intensities']) >= 18
        
        finally:
            Path(temp_file).unlink()


class TestMemoryUsageAndLeaks:
    """Test memory usage patterns and potential leaks."""
    
    @pytest.mark.asyncio
    async def test_cache_memory_cleanup(self):
        """Test that cache properly cleans up memory."""
        import gc
        
        cache = MemoryCache(max_size=1000, max_memory_mb=10)
        
        # Fill cache with large objects
        large_objects = []
        for i in range(500):
            large_obj = {"data": "x" * 10000, "id": i}  # ~10KB objects
            await cache.set(f"key_{i}", large_obj)
            if i < 100:  # Keep references to first 100
                large_objects.append(large_obj)
        
        initial_stats = cache.get_stats()
        
        # Clear cache
        await cache.clear()
        
        # Force garbage collection
        gc.collect()
        
        # Verify cleanup
        final_stats = cache.get_stats()
        assert final_stats['entries'] == 0
        assert final_stats['memory_usage_bytes'] == 0
        
        # Objects with references should still exist
        assert len(large_objects) == 100
    
    @pytest.mark.asyncio
    async def test_batch_processor_memory_efficiency(self):
        """Test batch processor doesn't accumulate memory over time."""
        processor = AsyncBatchProcessor(batch_size=10, max_concurrent=5)
        
        async def memory_intensive_task(item):
            # Create and immediately discard large object
            temp_data = [i for i in range(10000)]
            return sum(temp_data) + item
        
        # Process multiple batches
        for batch_num in range(10):
            items = list(range(batch_num * 100, (batch_num + 1) * 100))
            results = await processor.process_items(items, memory_intensive_task)
            
            # Verify results but don't keep references
            assert len(results) == 100
            del results  # Explicitly delete to help GC
        
        # Should complete without memory issues
        import gc
        gc.collect()  # Force cleanup
    
    @pytest.mark.asyncio
    async def test_monitor_callback_cleanup(self):
        """Test that monitor callbacks don't create memory leaks."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            sample_data = {
                "regions": {
                    "TEST_REGION": {
                        "historical": [
                            {
                                "timestamp": datetime.now().isoformat(),
                                "carbon_intensity": 100.0,
                                "renewable_percentage": 50.0,
                                "confidence": 0.9
                            }
                        ],
                        "forecast": []
                    }
                }
            }
            json.dump(sample_data, f)
            temp_file = f.name
        
        try:
            monitor = CarbonMonitor(
                regions=["TEST_REGION"],
                data_source=CarbonDataSource.CACHED,
                api_key=temp_file
            )
            
            callback_calls = []
            
            def test_callback(event_type, data):
                # Create some objects that reference the data
                callback_calls.append({
                    'event': event_type,
                    'data': data,
                    'timestamp': datetime.now(),
                    'large_data': [i for i in range(1000)]  # Create some memory usage
                })
            
            async with monitor:
                # Add and remove callbacks multiple times
                for i in range(10):
                    monitor.add_callback(test_callback)
                    await asyncio.sleep(0.1)
                    if i % 2 == 0:  # Remove every other time
                        monitor.remove_callback(test_callback)
                
                # Let it run briefly
                await monitor.start_monitoring()
                await asyncio.sleep(1)
                await monitor.stop_monitoring()
            
            # Callbacks should have been called but not cause memory issues
            assert len(callback_calls) > 0
            
            # Clear references
            callback_calls.clear()
            import gc
            gc.collect()
        
        finally:
            Path(temp_file).unlink()