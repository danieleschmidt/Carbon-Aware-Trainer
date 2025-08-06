"""Advanced metrics collection and export system."""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json

# Optional dependency handling
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

from .exceptions import MetricsError
from .types import CarbonIntensity, TrainingMetrics, TrainingState


logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Individual metric data point."""
    name: str
    value: Union[float, int, str]
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """System performance metrics."""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    gpu_utilization: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    power_draw_watts: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CarbonMetrics:
    """Carbon-related metrics."""
    current_intensity: float
    avg_intensity_1h: float
    avg_intensity_24h: float
    total_energy_kwh: float
    total_carbon_kg: float
    carbon_saved_kg: float
    renewable_percentage: float
    region: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TrainingProgressMetrics:
    """Training progress and efficiency metrics."""
    training_state: TrainingState
    epoch_number: Optional[int] = None
    batch_number: Optional[int] = None
    loss_value: Optional[float] = None
    accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
    training_efficiency: Optional[float] = None  # samples/second
    carbon_per_sample: Optional[float] = None  # gCO2 per sample
    pause_count: int = 0
    migration_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


class MetricsCollector:
    """Comprehensive metrics collection and aggregation system."""
    
    def __init__(self, collection_interval: int = 60, retention_hours: int = 168):
        """Initialize metrics collector.
        
        Args:
            collection_interval: Interval between collections in seconds
            retention_hours: How long to retain metrics in hours
        """
        self.collection_interval = collection_interval
        self.retention_hours = retention_hours
        self.retention_cutoff = timedelta(hours=retention_hours)
        
        # Metric storage
        self.metric_points: deque = deque(maxlen=100000)  # Raw metric points
        self.performance_metrics: deque = deque(maxlen=1000)
        self.carbon_metrics: deque = deque(maxlen=1000)
        self.training_metrics: deque = deque(maxlen=1000)
        
        # Aggregated metrics
        self.hourly_aggregates: Dict[str, Dict[datetime, Any]] = defaultdict(dict)
        self.daily_aggregates: Dict[str, Dict[datetime, Any]] = defaultdict(dict)
        
        # Custom metric collectors
        self.custom_collectors: Dict[str, Callable[[], Dict[str, Any]]] = {}
        
        # Collection tasks
        self._collection_task: Optional[asyncio.Task] = None
        self._aggregation_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Performance tracking
        self._last_collection_time = time.time()
        self._collection_durations: deque = deque(maxlen=100)
        
        logger.info("MetricsCollector initialized")
    
    async def start(self) -> None:
        """Start metrics collection."""
        if self._running:
            logger.warning("MetricsCollector already running")
            return
        
        self._running = True
        
        # Start collection tasks
        self._collection_task = asyncio.create_task(self._collection_worker())
        self._aggregation_task = asyncio.create_task(self._aggregation_worker())
        self._cleanup_task = asyncio.create_task(self._cleanup_worker())
        
        logger.info("MetricsCollector started")
    
    async def stop(self) -> None:
        """Stop metrics collection."""
        self._running = False
        
        # Cancel tasks
        for task in [self._collection_task, self._aggregation_task, self._cleanup_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("MetricsCollector stopped")
    
    def record_metric(
        self, 
        name: str, 
        value: Union[float, int, str],
        tags: Optional[Dict[str, str]] = None,
        unit: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Record a custom metric point.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for categorization
            unit: Optional unit of measurement
            timestamp: Optional timestamp (defaults to now)
        """
        metric_point = MetricPoint(
            name=name,
            value=value,
            timestamp=timestamp or datetime.now(),
            tags=tags or {},
            unit=unit
        )
        
        self.metric_points.append(metric_point)
        logger.debug(f"Recorded metric: {name}={value}")
    
    def record_performance_metrics(self, metrics: PerformanceMetrics) -> None:
        """Record system performance metrics.
        
        Args:
            metrics: Performance metrics to record
        """
        self.performance_metrics.append(metrics)
        
        # Also record as individual metric points for export
        tags = {'category': 'performance'}
        self.record_metric('system.cpu_percent', metrics.cpu_percent, tags, '%')
        self.record_metric('system.memory_percent', metrics.memory_percent, tags, '%')
        self.record_metric('system.disk_percent', metrics.disk_percent, tags, '%')
        self.record_metric('system.network_bytes_sent', metrics.network_bytes_sent, tags, 'bytes')
        self.record_metric('system.network_bytes_recv', metrics.network_bytes_recv, tags, 'bytes')
        
        if metrics.gpu_utilization is not None:
            self.record_metric('system.gpu_utilization', metrics.gpu_utilization, tags, '%')
        
        if metrics.power_draw_watts is not None:
            self.record_metric('system.power_draw_watts', metrics.power_draw_watts, tags, 'W')
    
    def record_carbon_metrics(self, metrics: CarbonMetrics) -> None:
        """Record carbon-related metrics.
        
        Args:
            metrics: Carbon metrics to record
        """
        self.carbon_metrics.append(metrics)
        
        # Record as individual metric points
        tags = {'category': 'carbon', 'region': metrics.region}
        self.record_metric('carbon.current_intensity', metrics.current_intensity, tags, 'gCO2/kWh')
        self.record_metric('carbon.avg_intensity_1h', metrics.avg_intensity_1h, tags, 'gCO2/kWh')
        self.record_metric('carbon.total_energy_kwh', metrics.total_energy_kwh, tags, 'kWh')
        self.record_metric('carbon.total_carbon_kg', metrics.total_carbon_kg, tags, 'kg')
        self.record_metric('carbon.renewable_percentage', metrics.renewable_percentage, tags, '%')
    
    def record_training_metrics(self, metrics: TrainingProgressMetrics) -> None:
        """Record training progress metrics.
        
        Args:
            metrics: Training metrics to record
        """
        self.training_metrics.append(metrics)
        
        # Record as individual metric points
        tags = {'category': 'training', 'state': metrics.training_state.value}
        
        if metrics.epoch_number is not None:
            self.record_metric('training.epoch', metrics.epoch_number, tags)
        
        if metrics.loss_value is not None:
            self.record_metric('training.loss', metrics.loss_value, tags)
        
        if metrics.accuracy is not None:
            self.record_metric('training.accuracy', metrics.accuracy, tags, '%')
        
        if metrics.training_efficiency is not None:
            self.record_metric('training.efficiency', metrics.training_efficiency, tags, 'samples/sec')
        
        self.record_metric('training.pause_count', metrics.pause_count, tags)
        self.record_metric('training.migration_count', metrics.migration_count, tags)
    
    def add_custom_collector(self, name: str, collector_func: Callable[[], Dict[str, Any]]) -> None:
        """Add custom metrics collector function.
        
        Args:
            name: Collector name
            collector_func: Function that returns dict of metric_name -> value
        """
        self.custom_collectors[name] = collector_func
        logger.info(f"Added custom metric collector: {name}")
    
    def remove_custom_collector(self, name: str) -> None:
        """Remove custom metrics collector.
        
        Args:
            name: Collector name
        """
        self.custom_collectors.pop(name, None)
        logger.info(f"Removed custom metric collector: {name}")
    
    def get_latest_performance_metrics(self) -> Optional[PerformanceMetrics]:
        """Get latest performance metrics.
        
        Returns:
            Latest performance metrics or None if unavailable
        """
        if self.performance_metrics:
            return self.performance_metrics[-1]
        return None
    
    def get_latest_carbon_metrics(self) -> Optional[CarbonMetrics]:
        """Get latest carbon metrics.
        
        Returns:
            Latest carbon metrics or None if unavailable
        """
        if self.carbon_metrics:
            return self.carbon_metrics[-1]
        return None
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary for specified time period.
        
        Args:
            hours: Number of hours to summarize
            
        Returns:
            Dictionary with metrics summary
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter metrics by time
        recent_performance = [m for m in self.performance_metrics if m.timestamp >= cutoff_time]
        recent_carbon = [m for m in self.carbon_metrics if m.timestamp >= cutoff_time]
        recent_training = [m for m in self.training_metrics if m.timestamp >= cutoff_time]
        recent_points = [p for p in self.metric_points if p.timestamp >= cutoff_time]
        
        summary = {
            'period_hours': hours,
            'data_points_collected': len(recent_points),
            'performance_metrics': self._summarize_performance_metrics(recent_performance),
            'carbon_metrics': self._summarize_carbon_metrics(recent_carbon),
            'training_metrics': self._summarize_training_metrics(recent_training),
            'collection_efficiency': self._get_collection_efficiency()
        }
        
        return summary
    
    def export_metrics(
        self, 
        format: str = 'prometheus',
        time_range: Optional[timedelta] = None
    ) -> str:
        """Export metrics in specified format.
        
        Args:
            format: Export format ('prometheus', 'json', 'csv')
            time_range: Optional time range to export
            
        Returns:
            Exported metrics as string
        """
        if time_range:
            cutoff_time = datetime.now() - time_range
            metrics_to_export = [p for p in self.metric_points if p.timestamp >= cutoff_time]
        else:
            metrics_to_export = list(self.metric_points)
        
        if format.lower() == 'prometheus':
            return self._export_prometheus_format(metrics_to_export)
        elif format.lower() == 'json':
            return self._export_json_format(metrics_to_export)
        elif format.lower() == 'csv':
            return self._export_csv_format(metrics_to_export)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    async def _collection_worker(self) -> None:
        """Background worker for metrics collection."""
        while self._running:
            try:
                collection_start = time.time()
                
                # Collect system performance metrics
                await self._collect_system_metrics()
                
                # Run custom collectors
                await self._run_custom_collectors()
                
                # Track collection performance
                collection_duration = time.time() - collection_start
                self._collection_durations.append(collection_duration)
                
                logger.debug(f"Metrics collection completed in {collection_duration:.3f}s")
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(min(self.collection_interval, 60))
    
    async def _collect_system_metrics(self) -> None:
        """Collect system performance metrics."""
        try:
            if not HAS_PSUTIL:
                logger.debug("psutil not available, skipping system metrics collection")
                # Create minimal fallback metrics
                perf_metrics = PerformanceMetrics(
                    cpu_percent=0.0,
                    memory_percent=0.0,
                    disk_percent=0.0,
                    network_bytes_sent=0,
                    network_bytes_recv=0,
                    gpu_utilization=None,
                    gpu_memory_percent=None,
                    power_draw_watts=None
                )
                self.record_performance_metrics(perf_metrics)
                return
            
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O
            net_io = psutil.net_io_counters()
            
            # Try to get GPU metrics
            gpu_util = None
            gpu_memory = None
            power_draw = None
            
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_util = sum(gpu.load for gpu in gpus) / len(gpus) * 100
                    gpu_memory = sum(gpu.memoryUtil for gpu in gpus) / len(gpus) * 100
                    # Power draw would need additional libraries or nvidia-ml-py
            except ImportError:
                pass
            
            # Create performance metrics
            perf_metrics = PerformanceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=disk.percent,
                network_bytes_sent=net_io.bytes_sent,
                network_bytes_recv=net_io.bytes_recv,
                gpu_utilization=gpu_util,
                gpu_memory_percent=gpu_memory,
                power_draw_watts=power_draw
            )
            
            self.record_performance_metrics(perf_metrics)
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    async def _run_custom_collectors(self) -> None:
        """Run all custom metric collectors."""
        for collector_name, collector_func in self.custom_collectors.items():
            try:
                start_time = time.time()
                
                # Run collector (handle both sync and async)
                if asyncio.iscoroutinefunction(collector_func):
                    metrics = await collector_func()
                else:
                    metrics = collector_func()
                
                # Record collected metrics
                tags = {'collector': collector_name}
                for metric_name, value in metrics.items():
                    self.record_metric(f"custom.{metric_name}", value, tags)
                
                collection_time = time.time() - start_time
                self.record_metric(
                    'metrics.collection_duration',
                    collection_time,
                    {'collector': collector_name},
                    's'
                )
                
            except Exception as e:
                logger.error(f"Custom collector {collector_name} failed: {e}")
    
    async def _aggregation_worker(self) -> None:
        """Background worker for metrics aggregation."""
        while self._running:
            try:
                await self._aggregate_metrics()
                await asyncio.sleep(3600)  # Run hourly
                
            except Exception as e:
                logger.error(f"Error in metrics aggregation: {e}")
                await asyncio.sleep(3600)
    
    async def _cleanup_worker(self) -> None:
        """Background worker for cleaning up old metrics."""
        while self._running:
            try:
                await self._cleanup_old_metrics()
                await asyncio.sleep(3600)  # Run hourly
                
            except Exception as e:
                logger.error(f"Error in metrics cleanup: {e}")
                await asyncio.sleep(3600)
    
    async def _aggregate_metrics(self) -> None:
        """Aggregate metrics into hourly and daily buckets."""
        cutoff_time = datetime.now() - self.retention_cutoff
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        current_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Group metrics by hour and day
        hourly_buckets = defaultdict(list)
        daily_buckets = defaultdict(list)
        
        for metric_point in self.metric_points:
            if metric_point.timestamp < cutoff_time:
                continue
            
            hour_bucket = metric_point.timestamp.replace(minute=0, second=0, microsecond=0)
            day_bucket = metric_point.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            
            hourly_buckets[hour_bucket].append(metric_point)
            daily_buckets[day_bucket].append(metric_point)
        
        # Aggregate hourly metrics
        for hour, points in hourly_buckets.items():
            if hour not in self.hourly_aggregates['summary']:
                self.hourly_aggregates['summary'][hour] = self._aggregate_metric_points(points)
        
        # Aggregate daily metrics
        for day, points in daily_buckets.items():
            if day not in self.daily_aggregates['summary']:
                self.daily_aggregates['summary'][day] = self._aggregate_metric_points(points)
        
        logger.debug("Metrics aggregation completed")
    
    async def _cleanup_old_metrics(self) -> None:
        """Clean up old metrics beyond retention period."""
        cutoff_time = datetime.now() - self.retention_cutoff
        
        # Clean up raw metric points
        initial_count = len(self.metric_points)
        self.metric_points = deque(
            (p for p in self.metric_points if p.timestamp >= cutoff_time),
            maxlen=self.metric_points.maxlen
        )
        cleaned_count = initial_count - len(self.metric_points)
        
        # Clean up typed metrics
        self.performance_metrics = deque(
            (m for m in self.performance_metrics if m.timestamp >= cutoff_time),
            maxlen=self.performance_metrics.maxlen
        )
        
        self.carbon_metrics = deque(
            (m for m in self.carbon_metrics if m.timestamp >= cutoff_time),
            maxlen=self.carbon_metrics.maxlen
        )
        
        self.training_metrics = deque(
            (m for m in self.training_metrics if m.timestamp >= cutoff_time),
            maxlen=self.training_metrics.maxlen
        )
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old metric points")
    
    def _aggregate_metric_points(self, points: List[MetricPoint]) -> Dict[str, Any]:
        """Aggregate a list of metric points."""
        if not points:
            return {}
        
        # Group by metric name
        metrics_by_name = defaultdict(list)
        for point in points:
            if isinstance(point.value, (int, float)):
                metrics_by_name[point.name].append(point.value)
        
        # Calculate aggregations
        aggregated = {}
        for metric_name, values in metrics_by_name.items():
            if values:
                aggregated[metric_name] = {
                    'count': len(values),
                    'sum': sum(values),
                    'avg': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'last': values[-1]
                }
        
        return aggregated
    
    def _summarize_performance_metrics(self, metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Summarize performance metrics."""
        if not metrics:
            return {}
        
        cpu_values = [m.cpu_percent for m in metrics]
        memory_values = [m.memory_percent for m in metrics]
        
        return {
            'avg_cpu_percent': sum(cpu_values) / len(cpu_values),
            'max_cpu_percent': max(cpu_values),
            'avg_memory_percent': sum(memory_values) / len(memory_values),
            'max_memory_percent': max(memory_values),
            'samples': len(metrics)
        }
    
    def _summarize_carbon_metrics(self, metrics: List[CarbonMetrics]) -> Dict[str, Any]:
        """Summarize carbon metrics."""
        if not metrics:
            return {}
        
        intensity_values = [m.current_intensity for m in metrics]
        
        return {
            'avg_carbon_intensity': sum(intensity_values) / len(intensity_values),
            'min_carbon_intensity': min(intensity_values),
            'max_carbon_intensity': max(intensity_values),
            'total_energy_kwh': metrics[-1].total_energy_kwh if metrics else 0,
            'total_carbon_kg': metrics[-1].total_carbon_kg if metrics else 0,
            'samples': len(metrics)
        }
    
    def _summarize_training_metrics(self, metrics: List[TrainingProgressMetrics]) -> Dict[str, Any]:
        """Summarize training metrics."""
        if not metrics:
            return {}
        
        states = [m.training_state for m in metrics]
        pause_counts = [m.pause_count for m in metrics]
        
        return {
            'training_states': list(set(s.value for s in states)),
            'total_pauses': max(pause_counts) if pause_counts else 0,
            'samples': len(metrics)
        }
    
    def _get_collection_efficiency(self) -> Dict[str, Any]:
        """Get metrics collection efficiency statistics."""
        if not self._collection_durations:
            return {}
        
        durations = list(self._collection_durations)
        
        return {
            'avg_collection_time_ms': (sum(durations) / len(durations)) * 1000,
            'max_collection_time_ms': max(durations) * 1000,
            'collections_completed': len(durations)
        }
    
    def _export_prometheus_format(self, metrics: List[MetricPoint]) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        # Group metrics by name
        metrics_by_name = defaultdict(list)
        for metric in metrics:
            if isinstance(metric.value, (int, float)):
                metrics_by_name[metric.name].append(metric)
        
        for metric_name, metric_list in metrics_by_name.items():
            # Add help and type comments
            safe_name = metric_name.replace('.', '_').replace('-', '_')
            lines.append(f"# HELP {safe_name} {metric_name}")
            lines.append(f"# TYPE {safe_name} gauge")
            
            # Add metric lines
            for metric in metric_list:
                tags_str = ""
                if metric.tags:
                    tag_pairs = [f'{k}="{v}"' for k, v in metric.tags.items()]
                    tags_str = "{" + ",".join(tag_pairs) + "}"
                
                timestamp_ms = int(metric.timestamp.timestamp() * 1000)
                lines.append(f"{safe_name}{tags_str} {metric.value} {timestamp_ms}")
        
        return "\n".join(lines)
    
    def _export_json_format(self, metrics: List[MetricPoint]) -> str:
        """Export metrics in JSON format."""
        metrics_data = []
        
        for metric in metrics:
            metrics_data.append({
                'name': metric.name,
                'value': metric.value,
                'timestamp': metric.timestamp.isoformat(),
                'tags': metric.tags,
                'unit': metric.unit
            })
        
        return json.dumps(metrics_data, indent=2)
    
    def _export_csv_format(self, metrics: List[MetricPoint]) -> str:
        """Export metrics in CSV format."""
        lines = ['name,value,timestamp,tags,unit']
        
        for metric in metrics:
            tags_json = json.dumps(metric.tags) if metric.tags else ""
            lines.append(f'"{metric.name}",{metric.value},{metric.timestamp.isoformat()},"{tags_json}","{metric.unit or ""}"')
        
        return "\n".join(lines)


# Global metrics collector instance
metrics_collector = MetricsCollector()