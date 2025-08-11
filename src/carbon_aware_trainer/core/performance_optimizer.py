"""
Advanced performance optimization engine with intelligent resource management.
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

import numpy as np

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    THROUGHPUT = "throughput"      # Maximize data processing throughput
    LATENCY = "latency"           # Minimize response latency
    EFFICIENCY = "efficiency"      # Optimize resource utilization
    BALANCED = "balanced"         # Balance all factors


class ResourceType(Enum):
    """Types of system resources."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    NETWORK = "network"
    DISK = "disk"


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""
    timestamp: datetime
    throughput_ops_sec: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    gpu_utilization: float = 0.0
    network_utilization: float = 0.0
    error_rate: float = 0.0
    queue_depth: int = 0
    active_threads: int = 0


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation."""
    component: str
    action: str
    impact_estimate: float  # Expected improvement (0-1)
    resource_cost: float    # Resource cost (0-1)
    confidence: float       # Confidence in recommendation (0-1)
    rationale: str
    parameters: Dict[str, Any] = field(default_factory=dict)


class AdaptiveResourceManager:
    """Manages system resources adaptively based on workload."""
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        max_memory_gb: Optional[float] = None,
        enable_gpu: bool = True
    ):
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.max_memory_gb = max_memory_gb or 16.0
        self.enable_gpu = enable_gpu
        
        # Thread pools
        self._thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self._process_pool = ProcessPoolExecutor(max_workers=min(self.max_workers, 4))
        
        # Resource tracking
        self.active_tasks = 0
        self.peak_tasks = 0
        self.total_completed = 0
        self.total_failed = 0
        
        # Performance history
        self.metrics_history: List[PerformanceMetrics] = []
        self.max_history = 1000
        
        # Adaptive parameters
        self.current_batch_size = 32
        self.min_batch_size = 8
        self.max_batch_size = 512
        self.queue_size_limit = 1000
        
    async def submit_task(
        self,
        func: Callable,
        *args,
        priority: int = 0,
        use_process_pool: bool = False,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Any:
        """Submit task for optimized execution."""
        start_time = time.time()
        self.active_tasks += 1
        self.peak_tasks = max(self.peak_tasks, self.active_tasks)
        
        try:
            if use_process_pool:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self._process_pool, func, *args, **kwargs
                )
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self._thread_pool, func, *args, **kwargs
                )
                
            self.total_completed += 1
            execution_time = time.time() - start_time
            
            # Record performance metrics
            await self._record_task_performance(execution_time, success=True)
            
            return result
            
        except Exception as e:
            self.total_failed += 1
            execution_time = time.time() - start_time
            await self._record_task_performance(execution_time, success=False)
            raise
            
        finally:
            self.active_tasks -= 1
            
    async def submit_batch(
        self,
        func: Callable,
        items: List[Any],
        batch_size: Optional[int] = None
    ) -> List[Any]:
        """Submit batch of items for optimized parallel processing."""
        if not items:
            return []
            
        # Use adaptive batch size if not specified
        effective_batch_size = batch_size or self.current_batch_size
        
        # Split items into batches
        batches = [
            items[i:i + effective_batch_size]
            for i in range(0, len(items), effective_batch_size)
        ]
        
        start_time = time.time()
        
        # Process batches concurrently
        tasks = []
        for batch in batches:
            task = self.submit_task(func, batch)
            tasks.append(task)
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results and handle exceptions
        flattened_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing error: {result}")
            elif isinstance(result, list):
                flattened_results.extend(result)
            else:
                flattened_results.append(result)
                
        processing_time = time.time() - start_time
        throughput = len(items) / processing_time if processing_time > 0 else 0
        
        # Adapt batch size based on performance
        await self._adapt_batch_size(throughput, processing_time)
        
        return flattened_results
        
    async def _record_task_performance(
        self,
        execution_time: float,
        success: bool
    ) -> None:
        """Record task performance metrics."""
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            throughput_ops_sec=1.0 / execution_time if execution_time > 0 else 0,
            latency_p50_ms=execution_time * 1000,
            cpu_utilization=0.0,  # Would be filled by system monitor
            error_rate=0.0 if success else 1.0,
            active_threads=self.active_tasks
        )
        
        self.metrics_history.append(metrics)
        
        # Keep history bounded
        if len(self.metrics_history) > self.max_history:
            self.metrics_history = self.metrics_history[-self.max_history:]
            
    async def _adapt_batch_size(
        self,
        throughput: float,
        processing_time: float
    ) -> None:
        """Adapt batch size based on performance."""
        # Get recent performance history
        if len(self.metrics_history) < 5:
            return
            
        recent_metrics = self.metrics_history[-5:]
        avg_latency = np.mean([m.latency_p50_ms for m in recent_metrics])
        
        # Increase batch size if latency is low and throughput could be higher
        if avg_latency < 100 and self.current_batch_size < self.max_batch_size:
            self.current_batch_size = min(
                self.current_batch_size * 1.2,
                self.max_batch_size
            )
            logger.debug(f"Increased batch size to {self.current_batch_size:.0f}")
            
        # Decrease batch size if latency is high
        elif avg_latency > 1000 and self.current_batch_size > self.min_batch_size:
            self.current_batch_size = max(
                self.current_batch_size * 0.8,
                self.min_batch_size
            )
            logger.debug(f"Decreased batch size to {self.current_batch_size:.0f}")
            
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.metrics_history:
            return {"status": "no_data"}
            
        recent_metrics = self.metrics_history[-10:]
        
        return {
            "active_tasks": self.active_tasks,
            "peak_tasks": self.peak_tasks,
            "total_completed": self.total_completed,
            "total_failed": self.total_failed,
            "success_rate": self.total_completed / (self.total_completed + self.total_failed)
                           if (self.total_completed + self.total_failed) > 0 else 1.0,
            "current_batch_size": self.current_batch_size,
            "avg_throughput": np.mean([m.throughput_ops_sec for m in recent_metrics]),
            "avg_latency_ms": np.mean([m.latency_p50_ms for m in recent_metrics]),
            "metrics_history_length": len(self.metrics_history)
        }
        
    def shutdown(self):
        """Shutdown resource manager."""
        self._thread_pool.shutdown(wait=True)
        self._process_pool.shutdown(wait=True)


class IntelligentCacheManager:
    """Intelligent caching system with adaptive policies."""
    
    def __init__(
        self,
        max_memory_mb: int = 1024,
        default_ttl: int = 3600,
        enable_compression: bool = True
    ):
        self.max_memory_mb = max_memory_mb
        self.default_ttl = default_ttl
        self.enable_compression = enable_compression
        
        # Cache storage
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self._access_counts: Dict[str, int] = {}
        self._memory_usage = 0
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None
                
            entry = self._cache[key]
            
            # Check if expired
            if time.time() > entry['expires_at']:
                del self._cache[key]
                del self._access_times[key]
                del self._access_counts[key]
                self.misses += 1
                return None
                
            # Update access statistics
            self._access_times[key] = time.time()
            self._access_counts[key] = self._access_counts.get(key, 0) + 1
            self.hits += 1
            
            return entry['value']
            
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set item in cache."""
        with self._lock:
            expires_at = time.time() + (ttl or self.default_ttl)
            
            # Estimate memory usage (rough approximation)
            estimated_size = len(str(value)) if value else 0
            
            # Check memory limit
            if (self._memory_usage + estimated_size) / (1024 * 1024) > self.max_memory_mb:
                # Evict items to make space
                self._evict_items(estimated_size)
                
            # Store in cache
            self._cache[key] = {
                'value': value,
                'expires_at': expires_at,
                'size': estimated_size,
                'created_at': time.time()
            }
            
            self._access_times[key] = time.time()
            self._access_counts[key] = 1
            self._memory_usage += estimated_size
            
            return True
            
    def _evict_items(self, needed_space: int) -> None:
        """Evict items using LRU + LFU hybrid policy."""
        if not self._cache:
            return
            
        # Calculate scores for eviction (lower = evict first)
        scores = {}
        current_time = time.time()
        
        for key in self._cache:
            # Combine recency and frequency
            last_access = self._access_times.get(key, 0)
            access_count = self._access_counts.get(key, 1)
            
            # Recency score (0-1, higher = more recent)
            recency = max(0, 1 - (current_time - last_access) / 3600)
            
            # Frequency score (normalized)
            max_count = max(self._access_counts.values()) if self._access_counts else 1
            frequency = access_count / max_count
            
            # Combined score (weighted)
            scores[key] = 0.6 * recency + 0.4 * frequency
            
        # Sort by score (ascending) and evict
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k])
        
        freed_space = 0
        for key in sorted_keys:
            if freed_space >= needed_space:
                break
                
            entry = self._cache[key]
            freed_space += entry['size']
            
            del self._cache[key]
            del self._access_times[key]
            del self._access_counts[key]
            self._memory_usage -= entry['size']
            self.evictions += 1
            
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._access_counts.clear()
            self._memory_usage = 0
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                "entries": len(self._cache),
                "memory_usage_mb": self._memory_usage / (1024 * 1024),
                "memory_limit_mb": self.max_memory_mb,
                "memory_utilization": (self._memory_usage / (1024 * 1024)) / self.max_memory_mb,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate": hit_rate,
                "total_requests": total_requests
            }


class PerformanceOptimizer:
    """Advanced performance optimization engine."""
    
    def __init__(
        self,
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
        enable_auto_scaling: bool = True
    ):
        self.strategy = strategy
        self.enable_auto_scaling = enable_auto_scaling
        
        # Components
        self.resource_manager = AdaptiveResourceManager()
        self.cache_manager = IntelligentCacheManager()
        
        # Performance monitoring
        self.optimization_history: List[Dict[str, Any]] = []
        self.recommendations: List[OptimizationRecommendation] = []
        
        # Auto-scaling parameters
        self.scale_up_threshold = 0.8    # CPU/Memory threshold to scale up
        self.scale_down_threshold = 0.3  # Threshold to scale down
        self.scale_check_interval = 60   # Seconds between scaling checks
        
    async def optimize_workload(
        self,
        workload_func: Callable,
        data: List[Any],
        target_latency_ms: Optional[float] = None,
        target_throughput: Optional[float] = None
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Optimize workload execution."""
        start_time = time.time()
        
        # Determine optimal execution strategy
        execution_plan = await self._plan_execution(
            len(data), target_latency_ms, target_throughput
        )
        
        # Execute with caching if beneficial
        if execution_plan['use_cache']:
            results = await self._execute_with_cache(
                workload_func, data, execution_plan
            )
        else:
            results = await self.resource_manager.submit_batch(
                workload_func, data, execution_plan['batch_size']
            )
            
        execution_time = time.time() - start_time
        
        # Generate performance report
        performance_report = {
            "execution_time_ms": execution_time * 1000,
            "throughput_items_sec": len(data) / execution_time,
            "items_processed": len(data),
            "cache_hit_rate": self.cache_manager.get_stats()['hit_rate'],
            "resource_utilization": self.resource_manager.get_performance_stats(),
            "execution_plan": execution_plan,
            "recommendations": await self._generate_recommendations()
        }
        
        self.optimization_history.append({
            "timestamp": datetime.now().isoformat(),
            "performance": performance_report
        })
        
        return results, performance_report
        
    async def _plan_execution(
        self,
        data_size: int,
        target_latency_ms: Optional[float],
        target_throughput: Optional[float]
    ) -> Dict[str, Any]:
        """Plan optimal execution strategy."""
        # Get current system performance
        resource_stats = self.resource_manager.get_performance_stats()
        cache_stats = self.cache_manager.get_stats()
        
        # Determine batch size based on strategy
        if self.strategy == OptimizationStrategy.LATENCY:
            # Smaller batches for lower latency
            batch_size = min(32, data_size)
        elif self.strategy == OptimizationStrategy.THROUGHPUT:
            # Larger batches for higher throughput
            batch_size = min(256, data_size)
        elif self.strategy == OptimizationStrategy.EFFICIENCY:
            # Balance batch size with resource utilization
            batch_size = min(64, data_size)
        else:  # BALANCED
            batch_size = min(128, data_size)
            
        # Adjust based on current performance
        if resource_stats.get("avg_latency_ms", 0) > 500:
            batch_size = max(16, batch_size // 2)  # Reduce batch size
        elif resource_stats.get("active_tasks", 0) < 2:
            batch_size = min(512, batch_size * 2)  # Increase batch size
            
        return {
            "batch_size": batch_size,
            "use_cache": cache_stats['hit_rate'] > 0.1,  # Use cache if it's useful
            "parallel_workers": min(self.resource_manager.max_workers, max(1, data_size // batch_size)),
            "strategy": self.strategy.value
        }
        
    async def _execute_with_cache(
        self,
        workload_func: Callable,
        data: List[Any],
        execution_plan: Dict[str, Any]
    ) -> List[Any]:
        """Execute workload with intelligent caching."""
        results = []
        cache_misses = []
        cache_miss_indices = []
        
        # Check cache for each item
        for i, item in enumerate(data):
            cache_key = f"workload_{hash(str(item))}"
            cached_result = self.cache_manager.get(cache_key)
            
            if cached_result is not None:
                results.append(cached_result)
            else:
                cache_misses.append(item)
                cache_miss_indices.append(i)
                results.append(None)  # Placeholder
                
        # Process cache misses
        if cache_misses:
            miss_results = await self.resource_manager.submit_batch(
                workload_func, cache_misses, execution_plan['batch_size']
            )
            
            # Update results and cache
            for idx, result in zip(cache_miss_indices, miss_results):
                results[idx] = result
                cache_key = f"workload_{hash(str(data[idx]))}"
                self.cache_manager.set(cache_key, result)
                
        return results
        
    async def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Analyze resource utilization
        resource_stats = self.resource_manager.get_performance_stats()
        cache_stats = self.cache_manager.get_stats()
        
        # Cache optimization recommendations
        if cache_stats['hit_rate'] < 0.2:
            recommendations.append({
                "type": "cache_tuning",
                "priority": "medium",
                "message": f"Low cache hit rate ({cache_stats['hit_rate']:.1%})",
                "suggestion": "Consider increasing cache TTL or improving cache key strategy"
            })
            
        elif cache_stats['memory_utilization'] > 0.9:
            recommendations.append({
                "type": "cache_memory",
                "priority": "high",
                "message": f"Cache memory usage high ({cache_stats['memory_utilization']:.1%})",
                "suggestion": "Increase cache memory limit or improve eviction policy"
            })
            
        # Resource utilization recommendations
        if resource_stats.get('success_rate', 1.0) < 0.95:
            recommendations.append({
                "type": "reliability",
                "priority": "high",
                "message": f"Low success rate ({resource_stats.get('success_rate', 0):.1%})",
                "suggestion": "Investigate failures and improve error handling"
            })
            
        # Latency recommendations
        avg_latency = resource_stats.get('avg_latency_ms', 0)
        if avg_latency > 1000:
            recommendations.append({
                "type": "performance",
                "priority": "medium",
                "message": f"High average latency ({avg_latency:.0f}ms)",
                "suggestion": "Consider reducing batch sizes or optimizing workload function"
            })
            
        return recommendations
        
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization summary."""
        return {
            "strategy": self.strategy.value,
            "resource_manager": self.resource_manager.get_performance_stats(),
            "cache_manager": self.cache_manager.get_stats(),
            "optimization_runs": len(self.optimization_history),
            "recent_recommendations": self.recommendations[-5:] if self.recommendations else [],
            "auto_scaling_enabled": self.enable_auto_scaling
        }
        
    def cleanup(self):
        """Clean up optimizer resources."""
        self.resource_manager.shutdown()
        self.cache_manager.clear()


# Global performance optimizer
performance_optimizer = PerformanceOptimizer()