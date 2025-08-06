"""Concurrent processing engine for high-performance carbon-aware training."""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
from collections import defaultdict, deque
import weakref
import math

from .types import CarbonIntensity, TrainingState
from .exceptions import CarbonDataError, SchedulingError


logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskState(Enum):
    """Task execution states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ConcurrentTask:
    """Task for concurrent execution."""
    task_id: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    region: Optional[str] = None
    carbon_threshold: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    state: TaskState = TaskState.PENDING
    result: Any = None
    error: Optional[Exception] = None
    dependencies: Set[str] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)


@dataclass
class WorkerStats:
    """Worker thread/process statistics."""
    worker_id: str
    worker_type: str  # 'thread' or 'process'
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0.0
    average_task_time: float = 0.0
    current_task: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    carbon_savings: float = 0.0


class ConcurrentExecutor:
    """High-performance concurrent task executor with carbon awareness."""
    
    def __init__(
        self,
        max_workers: int = None,
        thread_pool_size: int = None,
        process_pool_size: int = None,
        enable_process_pool: bool = True,
        queue_size: int = 1000,
        carbon_threshold: float = 100.0,
        adaptive_concurrency: bool = True,
        min_workers: int = 2,
        max_concurrent_tasks: int = 500
    ):
        """Initialize concurrent executor.
        
        Args:
            max_workers: Maximum total workers (auto-calculated if None)
            thread_pool_size: Thread pool size
            process_pool_size: Process pool size
            enable_process_pool: Enable process-based execution
            queue_size: Maximum task queue size
            carbon_threshold: Carbon intensity threshold for throttling
            adaptive_concurrency: Enable adaptive concurrency control
            min_workers: Minimum number of workers
            max_concurrent_tasks: Maximum concurrent tasks
        """
        import multiprocessing
        
        # Calculate optimal worker counts
        cpu_count = multiprocessing.cpu_count()
        self.max_workers = max_workers or min(32, (cpu_count + 4))
        self.thread_pool_size = thread_pool_size or max(4, cpu_count)
        self.process_pool_size = process_pool_size or max(2, cpu_count // 2)
        self.enable_process_pool = enable_process_pool
        self.queue_size = queue_size
        self.carbon_threshold = carbon_threshold
        self.adaptive_concurrency = adaptive_concurrency
        self.min_workers = min_workers
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Execution pools
        self._thread_pool: Optional[ThreadPoolExecutor] = None
        self._process_pool: Optional[ProcessPoolExecutor] = None
        
        # Task management
        self._task_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self._priority_queues = {
            TaskPriority.CRITICAL: asyncio.Queue(),
            TaskPriority.HIGH: asyncio.Queue(),
            TaskPriority.NORMAL: asyncio.Queue(),
            TaskPriority.LOW: asyncio.Queue()
        }
        self._active_tasks: Dict[str, ConcurrentTask] = {}
        self._completed_tasks: Dict[str, ConcurrentTask] = {}
        self._failed_tasks: Dict[str, ConcurrentTask] = {}
        
        # Worker management
        self._workers: List[asyncio.Task] = []
        self._worker_stats: Dict[str, WorkerStats] = {}
        self._current_concurrency = self.min_workers
        
        # Carbon awareness
        self._carbon_monitor: Optional[Any] = None
        self._current_carbon_intensity = 0.0
        self._carbon_throttling_active = False
        
        # Performance tracking
        self._performance_metrics = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'tasks_cancelled': 0,
            'total_execution_time': 0.0,
            'average_queue_time': 0.0,
            'throughput_per_second': 0.0,
            'carbon_throttling_events': 0,
            'concurrency_adjustments': 0
        }
        
        # Callbacks
        self._task_callbacks: List[Callable] = []
        self._performance_callbacks: List[Callable] = []
        
        # Control flags
        self._running = False
        self._shutdown_event = asyncio.Event()
        
        logger.info(f"Concurrent executor initialized (threads: {self.thread_pool_size}, "
                   f"processes: {self.process_pool_size if enable_process_pool else 0})")
    
    async def start(self) -> None:
        """Start the concurrent executor."""
        if self._running:
            return
        
        self._running = True
        self._shutdown_event.clear()
        
        # Initialize thread pool
        self._thread_pool = ThreadPoolExecutor(
            max_workers=self.thread_pool_size,
            thread_name_prefix="carbon_worker"
        )
        
        # Initialize process pool if enabled
        if self.enable_process_pool:
            try:
                self._process_pool = ProcessPoolExecutor(
                    max_workers=self.process_pool_size
                )
            except Exception as e:
                logger.warning(f"Failed to initialize process pool: {e}")
                self.enable_process_pool = False
        
        # Start worker tasks
        for i in range(self._current_concurrency):
            worker_task = asyncio.create_task(self._worker_loop(f"worker_{i}"))
            self._workers.append(worker_task)
            
            self._worker_stats[f"worker_{i}"] = WorkerStats(
                worker_id=f"worker_{i}",
                worker_type="async"
            )
        
        # Start performance monitoring
        asyncio.create_task(self._performance_monitor_loop())
        
        # Start adaptive concurrency adjustment
        if self.adaptive_concurrency:
            asyncio.create_task(self._adaptive_concurrency_loop())
        
        logger.info(f"Concurrent executor started with {len(self._workers)} workers")
    
    async def stop(self) -> None:
        """Stop the concurrent executor."""
        if not self._running:
            return
        
        self._running = False
        self._shutdown_event.set()
        
        # Cancel all active tasks
        for task in self._active_tasks.values():
            if task.state == TaskState.RUNNING:
                task.state = TaskState.CANCELLED
                self._performance_metrics['tasks_cancelled'] += 1
        
        # Wait for workers to finish
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        
        # Shutdown executor pools
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
        
        if self._process_pool:
            self._process_pool.shutdown(wait=True)
        
        logger.info("Concurrent executor stopped")
    
    async def submit_task(
        self,
        func: Callable,
        *args,
        task_id: Optional[str] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[float] = None,
        max_retries: int = 3,
        region: Optional[str] = None,
        carbon_threshold: Optional[float] = None,
        dependencies: Optional[Set[str]] = None,
        tags: Optional[Set[str]] = None,
        use_process_pool: bool = False,
        **kwargs
    ) -> str:
        """Submit task for concurrent execution.
        
        Args:
            func: Function to execute
            *args: Function arguments
            task_id: Optional task ID
            priority: Task priority
            timeout: Execution timeout
            max_retries: Maximum retry attempts
            region: Region for carbon-aware execution
            carbon_threshold: Carbon intensity threshold
            dependencies: Task dependencies
            tags: Task tags
            use_process_pool: Use process pool for execution
            **kwargs: Function keyword arguments
        
        Returns:
            Task ID
        """
        if not self._running:
            raise RuntimeError("Executor not running")
        
        # Generate task ID if not provided
        if task_id is None:
            task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Create task
        task = ConcurrentTask(
            task_id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout,
            max_retries=max_retries,
            region=region,
            carbon_threshold=carbon_threshold or self.carbon_threshold,
            dependencies=dependencies or set(),
            tags=tags or set()
        )
        
        # Add process pool flag to kwargs
        task.kwargs['_use_process_pool'] = use_process_pool
        
        # Queue task by priority
        await self._priority_queues[priority].put(task)
        self._performance_metrics['tasks_submitted'] += 1
        
        logger.debug(f"Submitted task {task_id} with priority {priority.name}")
        return task_id
    
    async def submit_batch(
        self,
        tasks: List[Dict[str, Any]],
        batch_priority: TaskPriority = TaskPriority.NORMAL,
        max_concurrent: Optional[int] = None
    ) -> List[str]:
        """Submit batch of tasks for execution.
        
        Args:
            tasks: List of task specifications
            batch_priority: Priority for all tasks in batch
            max_concurrent: Maximum concurrent tasks from this batch
        
        Returns:
            List of task IDs
        """
        task_ids = []
        semaphore = asyncio.Semaphore(max_concurrent or len(tasks))
        
        async def submit_single_task(task_spec):
            async with semaphore:
                task_id = await self.submit_task(
                    priority=batch_priority,
                    **task_spec
                )
                task_ids.append(task_id)
        
        # Submit all tasks concurrently
        await asyncio.gather(*[
            submit_single_task(task_spec) 
            for task_spec in tasks
        ])
        
        logger.info(f"Submitted batch of {len(task_ids)} tasks")
        return task_ids
    
    async def wait_for_task(
        self,
        task_id: str,
        timeout: Optional[float] = None
    ) -> ConcurrentTask:
        """Wait for task completion.
        
        Args:
            task_id: Task ID to wait for
            timeout: Wait timeout in seconds
        
        Returns:
            Completed task
        """
        start_time = time.time()
        
        while self._running:
            # Check if task is completed
            if task_id in self._completed_tasks:
                return self._completed_tasks[task_id]
            
            if task_id in self._failed_tasks:
                return self._failed_tasks[task_id]
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise asyncio.TimeoutError(f"Task {task_id} did not complete within {timeout}s")
            
            await asyncio.sleep(0.1)
        
        raise RuntimeError("Executor not running")
    
    async def wait_for_batch(
        self,
        task_ids: List[str],
        timeout: Optional[float] = None,
        return_when: str = "ALL_COMPLETED"
    ) -> List[ConcurrentTask]:
        """Wait for batch of tasks to complete.
        
        Args:
            task_ids: List of task IDs
            timeout: Wait timeout in seconds
            return_when: When to return ('ALL_COMPLETED' or 'FIRST_COMPLETED')
        
        Returns:
            List of completed tasks
        """
        start_time = time.time()
        completed_tasks = []
        remaining_ids = set(task_ids)
        
        while self._running and remaining_ids:
            # Check completed tasks
            for task_id in list(remaining_ids):
                if task_id in self._completed_tasks:
                    completed_tasks.append(self._completed_tasks[task_id])
                    remaining_ids.remove(task_id)
                elif task_id in self._failed_tasks:
                    completed_tasks.append(self._failed_tasks[task_id])
                    remaining_ids.remove(task_id)
            
            # Check return condition
            if return_when == "FIRST_COMPLETED" and completed_tasks:
                break
            elif return_when == "ALL_COMPLETED" and not remaining_ids:
                break
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                break
            
            await asyncio.sleep(0.1)
        
        return completed_tasks
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task.
        
        Args:
            task_id: Task ID to cancel
        
        Returns:
            True if task was cancelled
        """
        # Check if task is active
        if task_id in self._active_tasks:
            task = self._active_tasks[task_id]
            task.state = TaskState.CANCELLED
            self._performance_metrics['tasks_cancelled'] += 1
            return True
        
        # Remove from priority queues
        for priority_queue in self._priority_queues.values():
            temp_tasks = []
            
            # Drain queue and filter out cancelled task
            while not priority_queue.empty():
                try:
                    task = priority_queue.get_nowait()
                    if task.task_id != task_id:
                        temp_tasks.append(task)
                    else:
                        task.state = TaskState.CANCELLED
                        self._performance_metrics['tasks_cancelled'] += 1
                        
                        # Re-add other tasks
                        for temp_task in temp_tasks:
                            await priority_queue.put(temp_task)
                        return True
                except asyncio.QueueEmpty:
                    break
            
            # Re-add tasks if no match found
            for temp_task in temp_tasks:
                await priority_queue.put(temp_task)
        
        return False
    
    def get_task_status(self, task_id: str) -> Optional[ConcurrentTask]:
        """Get task status.
        
        Args:
            task_id: Task ID
        
        Returns:
            Task object or None if not found
        """
        # Check active tasks
        if task_id in self._active_tasks:
            return self._active_tasks[task_id]
        
        # Check completed tasks
        if task_id in self._completed_tasks:
            return self._completed_tasks[task_id]
        
        # Check failed tasks
        if task_id in self._failed_tasks:
            return self._failed_tasks[task_id]
        
        # Check priority queues
        for priority_queue in self._priority_queues.values():
            temp_tasks = []
            found_task = None
            
            # Temporarily drain queue to search
            while not priority_queue.empty():
                try:
                    task = priority_queue.get_nowait()
                    temp_tasks.append(task)
                    if task.task_id == task_id:
                        found_task = task
                except asyncio.QueueEmpty:
                    break
            
            # Restore queue
            asyncio.create_task(self._restore_queue(priority_queue, temp_tasks))
            
            if found_task:
                return found_task
        
        return None
    
    async def _restore_queue(self, queue: asyncio.Queue, tasks: List[ConcurrentTask]) -> None:
        """Restore tasks to queue."""
        for task in tasks:
            await queue.put(task)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        current_time = time.time()
        uptime = current_time - self._performance_metrics.get('start_time', current_time)
        
        # Calculate throughput
        if uptime > 0:
            throughput = self._performance_metrics['tasks_completed'] / uptime
        else:
            throughput = 0.0
        
        return {
            **self._performance_metrics,
            'active_tasks': len(self._active_tasks),
            'pending_tasks': sum(queue.qsize() for queue in self._priority_queues.values()),
            'current_concurrency': self._current_concurrency,
            'max_concurrency': self.max_workers,
            'throughput_per_second': throughput,
            'carbon_throttling_active': self._carbon_throttling_active,
            'current_carbon_intensity': self._current_carbon_intensity,
            'uptime_seconds': uptime,
            'worker_stats': {
                worker_id: {
                    'tasks_completed': stats.tasks_completed,
                    'tasks_failed': stats.tasks_failed,
                    'average_task_time': stats.average_task_time,
                    'current_task': stats.current_task,
                    'carbon_savings': stats.carbon_savings
                }
                for worker_id, stats in self._worker_stats.items()
            }
        }
    
    async def _worker_loop(self, worker_id: str) -> None:
        """Main worker loop."""
        worker_stats = self._worker_stats[worker_id]
        
        while self._running and not self._shutdown_event.is_set():
            try:
                # Get next task with priority
                task = await self._get_next_task()
                
                if task is None:
                    await asyncio.sleep(0.1)
                    continue
                
                # Check carbon throttling
                if self._should_throttle_for_carbon(task):
                    # Put task back in queue and wait
                    await self._priority_queues[task.priority].put(task)
                    await asyncio.sleep(1.0)
                    continue
                
                # Check dependencies
                if not self._dependencies_satisfied(task):
                    await self._priority_queues[task.priority].put(task)
                    await asyncio.sleep(0.5)
                    continue
                
                # Execute task
                await self._execute_task(task, worker_stats)
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1.0)
    
    async def _get_next_task(self) -> Optional[ConcurrentTask]:
        """Get next task from priority queues."""
        # Try queues in priority order
        for priority in [TaskPriority.CRITICAL, TaskPriority.HIGH, 
                        TaskPriority.NORMAL, TaskPriority.LOW]:
            queue = self._priority_queues[priority]
            
            if not queue.empty():
                try:
                    return queue.get_nowait()
                except asyncio.QueueEmpty:
                    continue
        
        return None
    
    def _should_throttle_for_carbon(self, task: ConcurrentTask) -> bool:
        """Check if task should be throttled due to carbon intensity."""
        if not task.carbon_threshold:
            return False
        
        if self._current_carbon_intensity > task.carbon_threshold:
            if not self._carbon_throttling_active:
                self._carbon_throttling_active = True
                self._performance_metrics['carbon_throttling_events'] += 1
                logger.info(f"Carbon throttling activated (intensity: {self._current_carbon_intensity})")
            return True
        
        if self._carbon_throttling_active:
            self._carbon_throttling_active = False
            logger.info("Carbon throttling deactivated")
        
        return False
    
    def _dependencies_satisfied(self, task: ConcurrentTask) -> bool:
        """Check if task dependencies are satisfied."""
        if not task.dependencies:
            return True
        
        # Check if all dependencies are completed
        for dep_id in task.dependencies:
            if dep_id not in self._completed_tasks:
                return False
        
        return True
    
    async def _execute_task(self, task: ConcurrentTask, worker_stats: WorkerStats) -> None:
        """Execute a task."""
        task.state = TaskState.RUNNING
        task.started_at = datetime.now()
        self._active_tasks[task.task_id] = task
        worker_stats.current_task = task.task_id
        
        start_time = time.time()
        
        try:
            # Choose execution method
            use_process_pool = task.kwargs.pop('_use_process_pool', False)
            
            if use_process_pool and self._process_pool:
                # Execute in process pool
                result = await self._execute_in_process_pool(task)
            elif asyncio.iscoroutinefunction(task.func):
                # Execute as coroutine
                if task.timeout:
                    result = await asyncio.wait_for(task.func(*task.args, **task.kwargs), task.timeout)
                else:
                    result = await task.func(*task.args, **task.kwargs)
            else:
                # Execute in thread pool
                result = await self._execute_in_thread_pool(task)
            
            # Task completed successfully
            execution_time = time.time() - start_time
            task.result = result
            task.state = TaskState.COMPLETED
            task.completed_at = datetime.now()
            
            # Update statistics
            worker_stats.tasks_completed += 1
            worker_stats.total_execution_time += execution_time
            worker_stats.average_task_time = (
                worker_stats.total_execution_time / worker_stats.tasks_completed
            )
            worker_stats.last_activity = datetime.now()
            
            self._performance_metrics['tasks_completed'] += 1
            self._performance_metrics['total_execution_time'] += execution_time
            
            # Move to completed tasks
            del self._active_tasks[task.task_id]
            self._completed_tasks[task.task_id] = task
            
            # Notify callbacks
            await self._notify_task_callbacks('completed', task)
            
            logger.debug(f"Task {task.task_id} completed in {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Handle task failure
            task.error = e
            task.state = TaskState.FAILED
            task.completed_at = datetime.now()
            
            # Update statistics
            worker_stats.tasks_failed += 1
            worker_stats.last_activity = datetime.now()
            self._performance_metrics['tasks_failed'] += 1
            
            # Retry if possible
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.state = TaskState.PENDING
                await self._priority_queues[task.priority].put(task)
                logger.warning(f"Task {task.task_id} failed, retrying ({task.retry_count}/{task.max_retries}): {e}")
            else:
                # Move to failed tasks
                del self._active_tasks[task.task_id]
                self._failed_tasks[task.task_id] = task
                
                # Notify callbacks
                await self._notify_task_callbacks('failed', task)
                
                logger.error(f"Task {task.task_id} failed permanently: {e}")
        
        finally:
            worker_stats.current_task = None
    
    async def _execute_in_thread_pool(self, task: ConcurrentTask) -> Any:
        """Execute task in thread pool."""
        loop = asyncio.get_event_loop()
        
        if task.timeout:
            future = self._thread_pool.submit(task.func, *task.args, **task.kwargs)
            return await asyncio.wait_for(
                loop.run_in_executor(None, future.result, task.timeout),
                task.timeout
            )
        else:
            return await loop.run_in_executor(
                self._thread_pool,
                task.func,
                *task.args,
                **task.kwargs
            )
    
    async def _execute_in_process_pool(self, task: ConcurrentTask) -> Any:
        """Execute task in process pool."""
        loop = asyncio.get_event_loop()
        
        if task.timeout:
            future = self._process_pool.submit(task.func, *task.args, **task.kwargs)
            return await asyncio.wait_for(
                loop.run_in_executor(None, future.result, task.timeout),
                task.timeout
            )
        else:
            return await loop.run_in_executor(
                self._process_pool,
                task.func,
                *task.args,
                **task.kwargs
            )
    
    async def _notify_task_callbacks(self, event: str, task: ConcurrentTask) -> None:
        """Notify task completion callbacks."""
        for callback in self._task_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event, task)
                else:
                    callback(event, task)
            except Exception as e:
                logger.error(f"Task callback error: {e}")
    
    async def _performance_monitor_loop(self) -> None:
        """Performance monitoring loop."""
        self._performance_metrics['start_time'] = time.time()
        
        while self._running and not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
                # Calculate performance metrics
                await self._update_performance_metrics()
                
                # Notify performance callbacks
                for callback in self._performance_callbacks:
                    try:
                        metrics = self.get_performance_metrics()
                        if asyncio.iscoroutinefunction(callback):
                            await callback(metrics)
                        else:
                            callback(metrics)
                    except Exception as e:
                        logger.error(f"Performance callback error: {e}")
                
            except Exception as e:
                logger.error(f"Performance monitor error: {e}")
    
    async def _update_performance_metrics(self) -> None:
        """Update performance metrics."""
        # Update throughput
        current_time = time.time()
        uptime = current_time - self._performance_metrics['start_time']
        
        if uptime > 0:
            self._performance_metrics['throughput_per_second'] = (
                self._performance_metrics['tasks_completed'] / uptime
            )
        
        # Log performance summary
        metrics = self.get_performance_metrics()
        logger.info(
            f"Performance: {metrics['tasks_completed']} completed, "
            f"{metrics['active_tasks']} active, "
            f"{metrics['throughput_per_second']:.1f} tasks/s, "
            f"concurrency: {metrics['current_concurrency']}"
        )
    
    async def _adaptive_concurrency_loop(self) -> None:
        """Adaptive concurrency adjustment loop."""
        while self._running and not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Adjust every minute
                
                await self._adjust_concurrency()
                
            except Exception as e:
                logger.error(f"Adaptive concurrency error: {e}")
    
    async def _adjust_concurrency(self) -> None:
        """Adjust concurrency based on performance metrics."""
        metrics = self.get_performance_metrics()
        
        # Calculate queue pressure
        total_pending = metrics['pending_tasks']
        queue_pressure = total_pending / max(1, metrics['current_concurrency'])
        
        # Calculate throughput trend
        current_throughput = metrics['throughput_per_second']
        
        # Adjustment logic
        if queue_pressure > 5 and self._current_concurrency < self.max_workers:
            # Scale up
            new_concurrency = min(
                self.max_workers,
                self._current_concurrency + max(1, int(self._current_concurrency * 0.2))
            )
            await self._scale_workers(new_concurrency)
            
        elif queue_pressure < 1 and self._current_concurrency > self.min_workers:
            # Scale down
            new_concurrency = max(
                self.min_workers,
                self._current_concurrency - max(1, int(self._current_concurrency * 0.1))
            )
            await self._scale_workers(new_concurrency)
    
    async def _scale_workers(self, target_concurrency: int) -> None:
        """Scale worker count."""
        current_count = len(self._workers)
        
        if target_concurrency > current_count:
            # Add workers
            for i in range(target_concurrency - current_count):
                worker_id = f"worker_{current_count + i}"
                worker_task = asyncio.create_task(self._worker_loop(worker_id))
                self._workers.append(worker_task)
                
                self._worker_stats[worker_id] = WorkerStats(
                    worker_id=worker_id,
                    worker_type="async"
                )
            
            logger.info(f"Scaled up to {target_concurrency} workers")
            
        elif target_concurrency < current_count:
            # Remove workers (they will exit naturally when queue is empty)
            workers_to_remove = current_count - target_concurrency
            
            for _ in range(workers_to_remove):
                if self._workers:
                    worker = self._workers.pop()
                    # Workers will exit when they see shutdown event or find no work
            
            logger.info(f"Scaled down to {target_concurrency} workers")
        
        self._current_concurrency = target_concurrency
        self._performance_metrics['concurrency_adjustments'] += 1
    
    def add_task_callback(self, callback: Callable) -> None:
        """Add task completion callback."""
        self._task_callbacks.append(callback)
    
    def add_performance_callback(self, callback: Callable) -> None:
        """Add performance monitoring callback."""
        self._performance_callbacks.append(callback)
    
    def set_carbon_monitor(self, monitor: Any) -> None:
        """Set carbon intensity monitor."""
        self._carbon_monitor = monitor
        
        # Add callback to monitor carbon changes
        if hasattr(monitor, 'add_callback'):
            monitor.add_callback(self._on_carbon_change)
    
    async def _on_carbon_change(self, event_type: str, data: Dict[str, Any]) -> None:
        """Handle carbon intensity changes."""
        if event_type == 'intensity_change':
            self._current_carbon_intensity = data.get('new_intensity', {}).get('carbon_intensity', 0)


# Global concurrent executor instance
_global_executor: Optional[ConcurrentExecutor] = None


def get_global_executor(config: Optional[Dict[str, Any]] = None) -> ConcurrentExecutor:
    """Get global concurrent executor instance."""
    global _global_executor
    
    if _global_executor is None:
        executor_config = config or {}
        _global_executor = ConcurrentExecutor(**executor_config)
    
    return _global_executor


# Convenient decorators
def concurrent(
    priority: TaskPriority = TaskPriority.NORMAL,
    timeout: Optional[float] = None,
    max_retries: int = 3,
    use_process_pool: bool = False,
    executor: Optional[ConcurrentExecutor] = None
):
    """Decorator to make function execution concurrent."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            executor_instance = executor or get_global_executor()
            
            task_id = await executor_instance.submit_task(
                func,
                *args,
                priority=priority,
                timeout=timeout,
                max_retries=max_retries,
                use_process_pool=use_process_pool,
                **kwargs
            )
            
            # Wait for completion and return result
            completed_task = await executor_instance.wait_for_task(task_id)
            
            if completed_task.state == TaskState.COMPLETED:
                return completed_task.result
            else:
                raise completed_task.error or RuntimeError(f"Task {task_id} failed")
        
        return wrapper
    return decorator