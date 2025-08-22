"""
Generation 3 Advanced Optimizer for Carbon-Aware Training
Optimization & Scaling Features
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from concurrent.futures import ThreadPoolExecutor
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class OptimizationMode(Enum):
    """Optimization strategies."""
    CARBON_FIRST = "carbon_first"
    PERFORMANCE_FIRST = "performance_first"
    BALANCED = "balanced"
    COST_OPTIMIZED = "cost_optimized"


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    gpu_utilization: float = 0.0
    throughput: float = 0.0  # steps per second


@dataclass
class PerformanceProfile:
    """Performance optimization profile."""
    name: str
    batch_size_multiplier: float = 1.0
    parallel_workers: int = 1
    memory_optimization: bool = False
    mixed_precision: bool = False


class Generation3Optimizer:
    """Advanced optimizer for Generation 3 features."""
    
    def __init__(
        self,
        optimization_mode: OptimizationMode = OptimizationMode.BALANCED,
        target_utilization: float = 0.8,
        max_workers: int = 4
    ):
        """Initialize Generation 3 optimizer.
        
        Args:
            optimization_mode: Optimization strategy
            target_utilization: Target resource utilization
            max_workers: Maximum parallel workers
        """
        self.optimization_mode = optimization_mode
        self.target_utilization = target_utilization
        self.max_workers = max_workers
        
        # State tracking
        self.resource_history: List[ResourceMetrics] = []
        self.current_profile = PerformanceProfile("default")
        self.baseline_throughput: Optional[float] = None
        
        # Thread pool for parallel operations
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Monitoring
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        logger.info(f"Generation3Optimizer initialized with {optimization_mode.value} mode")
    
    async def initialize(self) -> None:
        """Initialize optimizer."""
        # Start monitoring
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Establish baseline
        await self._establish_baseline()
        
        logger.info("Generation 3 optimizer initialized")
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = await self._collect_metrics()
                if metrics:
                    self.resource_history.append(metrics)
                    
                    # Keep recent history only
                    cutoff_time = datetime.now() - timedelta(hours=1)
                    self.resource_history = [
                        m for m in self.resource_history 
                        if m.timestamp >= cutoff_time
                    ]
                
                await asyncio.sleep(10.0)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(20.0)
    
    async def _collect_metrics(self) -> Optional[ResourceMetrics]:
        """Collect resource metrics."""
        try:
            # Try to get actual metrics
            try:
                import psutil
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
            except ImportError:
                # Fallback estimates
                cpu_percent = 50.0
                memory_percent = 60.0
            
            return ResourceMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                gpu_utilization=0.0,
                throughput=0.0
            )
            
        except Exception as e:
            logger.warning(f"Failed to collect metrics: {e}")
            return None
    
    async def _establish_baseline(self) -> None:
        """Establish performance baseline."""
        logger.info("Establishing performance baseline...")
        
        # Simple baseline measurement
        start_time = time.time()
        await asyncio.sleep(0.1)  # Simulate work
        duration = time.time() - start_time
        
        self.baseline_throughput = 1.0 / duration  # Simple throughput estimate
        
        logger.info(f"Baseline throughput: {self.baseline_throughput:.2f} ops/sec")
    
    def optimize_training_params(self, training_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize training parameters based on current profile."""
        optimized_kwargs = training_kwargs.copy()
        
        # Apply current profile optimizations
        if 'batch_size' in optimized_kwargs:
            original_size = optimized_kwargs['batch_size']
            optimized_size = int(original_size * self.current_profile.batch_size_multiplier)
            optimized_kwargs['batch_size'] = max(1, optimized_size)
        
        # Add optimization flags
        optimized_kwargs.update({
            'parallel_workers': self.current_profile.parallel_workers,
            'memory_optimization': self.current_profile.memory_optimization,
            'mixed_precision': self.current_profile.mixed_precision,
            'optimization_mode': self.optimization_mode.value
        })
        
        return optimized_kwargs
    
    async def optimize_batch_processing(
        self,
        batches: List[Any],
        process_fn: Callable,
        carbon_intensity: float = 100.0
    ) -> List[Any]:
        """Optimize batch processing with parallelization."""
        if not batches:
            return []
        
        # Adjust parallelization based on carbon intensity and mode
        if self.optimization_mode == OptimizationMode.CARBON_FIRST:
            # Lower parallelization for carbon efficiency
            max_parallel = max(1, self.current_profile.parallel_workers // 2)
        elif self.optimization_mode == OptimizationMode.PERFORMANCE_FIRST:
            # Maximum parallelization for speed
            max_parallel = min(self.max_workers, len(batches))
        else:
            # Balanced approach
            max_parallel = self.current_profile.parallel_workers
        
        # Adjust for carbon intensity
        if carbon_intensity > 200:
            max_parallel = max(1, max_parallel // 2)
        elif carbon_intensity < 50:
            max_parallel = min(self.max_workers, max_parallel * 2)
        
        # Process batches
        return await self._process_batches_parallel(batches, process_fn, max_parallel)
    
    async def _process_batches_parallel(
        self,
        batches: List[Any],
        process_fn: Callable,
        max_parallel: int
    ) -> List[Any]:
        """Process batches in parallel."""
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def process_with_limit(batch):
            async with semaphore:
                if asyncio.iscoroutinefunction(process_fn):
                    return await process_fn(batch)
                else:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(self.thread_pool, process_fn, batch)
        
        return await asyncio.gather(*[process_with_limit(batch) for batch in batches])
    
    def adapt_profile(self, carbon_intensity: float, resource_pressure: float = 0.5) -> None:
        """Adapt performance profile based on conditions."""
        if self.optimization_mode == OptimizationMode.CARBON_FIRST:
            # Prioritize carbon efficiency
            if carbon_intensity > 150:
                self.current_profile = PerformanceProfile(
                    name="carbon_optimized",
                    batch_size_multiplier=0.8,
                    parallel_workers=max(1, self.max_workers // 2),
                    memory_optimization=True,
                    mixed_precision=True
                )
            else:
                self.current_profile = PerformanceProfile(
                    name="carbon_balanced",
                    batch_size_multiplier=1.0,
                    parallel_workers=self.max_workers,
                    memory_optimization=True,
                    mixed_precision=False
                )
        
        elif self.optimization_mode == OptimizationMode.PERFORMANCE_FIRST:
            # Prioritize speed
            if resource_pressure < 0.7:
                self.current_profile = PerformanceProfile(
                    name="performance_max",
                    batch_size_multiplier=1.5,
                    parallel_workers=self.max_workers,
                    memory_optimization=False,
                    mixed_precision=False
                )
            else:
                self.current_profile = PerformanceProfile(
                    name="performance_efficient",
                    batch_size_multiplier=1.2,
                    parallel_workers=self.max_workers,
                    memory_optimization=True,
                    mixed_precision=True
                )
        
        else:  # BALANCED or COST_OPTIMIZED
            # Balanced approach
            multiplier = 1.2 if carbon_intensity < 100 else 0.9
            workers = self.max_workers if resource_pressure < 0.8 else max(1, self.max_workers // 2)
            
            self.current_profile = PerformanceProfile(
                name="balanced_adaptive",
                batch_size_multiplier=multiplier,
                parallel_workers=workers,
                memory_optimization=resource_pressure > 0.7,
                mixed_precision=carbon_intensity > 120
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        if not self.resource_history:
            return {
                'current_profile': self.current_profile.name,
                'optimization_mode': self.optimization_mode.value,
                'baseline_throughput': self.baseline_throughput,
                'monitoring_samples': 0
            }
        
        recent_metrics = self.resource_history[-10:]
        
        return {
            'current_profile': self.current_profile.name,
            'optimization_mode': self.optimization_mode.value,
            'baseline_throughput': self.baseline_throughput,
            'monitoring_samples': len(self.resource_history),
            'recent_cpu_avg': sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
            'recent_memory_avg': sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
            'profile_settings': {
                'batch_multiplier': self.current_profile.batch_size_multiplier,
                'parallel_workers': self.current_profile.parallel_workers,
                'memory_optimization': self.current_profile.memory_optimization,
                'mixed_precision': self.current_profile.mixed_precision
            }
        }
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.thread_pool.shutdown(wait=True)
        
        logger.info("Generation3Optimizer cleanup completed")