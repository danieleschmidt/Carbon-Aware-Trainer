"""Enhanced monitoring and alerting system for production deployments."""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(str, Enum):
    """Types of metrics collected."""
    SYSTEM = "system"
    CARBON = "carbon"
    TRAINING = "training"
    PERFORMANCE = "performance"
    BUSINESS = "business"


@dataclass
class Alert:
    """System alert representation."""
    id: str
    level: AlertLevel
    component: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False


@dataclass
class Metric:
    """System metric representation."""
    name: str
    value: Union[int, float]
    unit: str
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)


class EnhancedMonitoringSystem:
    """Production-grade monitoring and alerting system."""
    
    def __init__(
        self,
        metric_retention_hours: int = 24,
        alert_retention_hours: int = 168,  # 7 days
        alert_cooldown_minutes: int = 15
    ):
        """Initialize enhanced monitoring system.
        
        Args:
            metric_retention_hours: How long to keep metrics in memory
            alert_retention_hours: How long to keep alerts in memory
            alert_cooldown_minutes: Minimum time between similar alerts
        """
        self.metric_retention = timedelta(hours=metric_retention_hours)
        self.alert_retention = timedelta(hours=alert_retention_hours)
        self.alert_cooldown = timedelta(minutes=alert_cooldown_minutes)
        
        # Metrics storage
        self._metrics: Dict[str, List[Metric]] = {}
        self._metric_lock = asyncio.Lock()
        
        # Alerts storage
        self._alerts: List[Alert] = []
        self._alert_lock = asyncio.Lock()
        self._alert_callbacks: List[Callable[[Alert], None]] = []
        
        # System state tracking
        self._system_start_time = datetime.now()
        self._last_cleanup = datetime.now()
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self._request_times: List[float] = []
        self._error_counts: Dict[str, int] = {}
        
        # Thresholds (can be configured)
        self.thresholds = {
            'cpu_percent': {'warning': 70.0, 'critical': 90.0},
            'memory_percent': {'warning': 75.0, 'critical': 90.0},
            'disk_percent': {'warning': 80.0, 'critical': 95.0},
            'response_time_ms': {'warning': 5000.0, 'critical': 10000.0},
            'error_rate_percent': {'warning': 5.0, 'critical': 15.0},
            'carbon_intensity': {'warning': 300.0, 'critical': 500.0}
        }
        
        logger.info("Enhanced monitoring system initialized")
    
    async def start_monitoring(self) -> None:
        """Start continuous monitoring."""
        if self._monitoring_task and not self._monitoring_task.done():
            return
        
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started enhanced monitoring")
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring."""
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped enhanced monitoring")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while True:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Check thresholds and generate alerts
                await self._check_thresholds()
                
                # Cleanup old data
                await self._cleanup_old_data()
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)
    
    async def _collect_system_metrics(self) -> None:
        """Collect system performance metrics."""
        if not HAS_PSUTIL:
            return
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            await self.record_metric(
                "cpu_usage_percent",
                cpu_percent,
                "percent",
                MetricType.SYSTEM
            )
            
            # Memory metrics
            memory = psutil.virtual_memory()
            await self.record_metric(
                "memory_usage_percent",
                memory.percent,
                "percent",
                MetricType.SYSTEM
            )
            await self.record_metric(
                "memory_available_gb",
                memory.available / (1024**3),
                "GB",
                MetricType.SYSTEM
            )
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            await self.record_metric(
                "disk_usage_percent",
                disk_percent,
                "percent",
                MetricType.SYSTEM
            )
            
            # Network metrics (if available)
            try:
                net_io = psutil.net_io_counters()
                await self.record_metric(
                    "network_bytes_sent",
                    net_io.bytes_sent,
                    "bytes",
                    MetricType.SYSTEM
                )
                await self.record_metric(
                    "network_bytes_recv",
                    net_io.bytes_recv,
                    "bytes",
                    MetricType.SYSTEM
                )
            except Exception:
                pass  # Network stats may not be available
            
        except Exception as e:
            logger.error(f"System metrics collection error: {e}")
    
    async def record_metric(
        self,
        name: str,
        value: Union[int, float],
        unit: str,
        metric_type: MetricType,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
            metric_type: Type of metric
            tags: Optional tags for metric
        """
        metric = Metric(
            name=name,
            value=value,
            unit=unit,
            metric_type=metric_type,
            tags=tags or {}
        )
        
        async with self._metric_lock:
            if name not in self._metrics:
                self._metrics[name] = []
            
            self._metrics[name].append(metric)
            
            # Limit in-memory metrics to prevent unbounded growth
            max_metrics = 1440  # 24 hours at 1 minute intervals
            if len(self._metrics[name]) > max_metrics:
                self._metrics[name] = self._metrics[name][-max_metrics:]
    
    async def record_carbon_metrics(
        self,
        intensity: float,
        region: str,
        energy_kwh: float,
        carbon_kg: float
    ) -> None:
        """Record carbon-related metrics.
        
        Args:
            intensity: Carbon intensity in gCO2/kWh
            region: Region code
            energy_kwh: Energy consumption in kWh
            carbon_kg: Carbon emissions in kg CO2
        """
        tags = {"region": region}
        
        await self.record_metric(
            "carbon_intensity",
            intensity,
            "gCO2/kWh",
            MetricType.CARBON,
            tags
        )
        
        await self.record_metric(
            "energy_consumption",
            energy_kwh,
            "kWh",
            MetricType.CARBON,
            tags
        )
        
        await self.record_metric(
            "carbon_emissions",
            carbon_kg,
            "kg_CO2",
            MetricType.CARBON,
            tags
        )
    
    async def record_training_metrics(
        self,
        step: int,
        epoch: int,
        loss: float,
        accuracy: Optional[float] = None,
        learning_rate: Optional[float] = None
    ) -> None:
        """Record training-related metrics.
        
        Args:
            step: Training step
            epoch: Training epoch
            loss: Training loss
            accuracy: Model accuracy (if available)
            learning_rate: Current learning rate (if available)
        """
        await self.record_metric(
            "training_step",
            step,
            "step",
            MetricType.TRAINING
        )
        
        await self.record_metric(
            "training_epoch",
            epoch,
            "epoch",
            MetricType.TRAINING
        )
        
        await self.record_metric(
            "training_loss",
            loss,
            "loss",
            MetricType.TRAINING
        )
        
        if accuracy is not None:
            await self.record_metric(
                "training_accuracy",
                accuracy,
                "percent",
                MetricType.TRAINING
            )
        
        if learning_rate is not None:
            await self.record_metric(
                "learning_rate",
                learning_rate,
                "rate",
                MetricType.TRAINING
            )
    
    async def record_request_time(self, duration_ms: float) -> None:
        """Record request processing time."""
        self._request_times.append(duration_ms)
        
        # Keep only recent request times
        if len(self._request_times) > 1000:
            self._request_times = self._request_times[-1000:]
        
        await self.record_metric(
            "request_duration",
            duration_ms,
            "ms",
            MetricType.PERFORMANCE
        )
    
    async def record_error(self, error_type: str, details: Optional[str] = None) -> None:
        """Record error occurrence."""
        self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1
        
        await self.record_metric(
            f"error_count_{error_type}",
            self._error_counts[error_type],
            "count",
            MetricType.PERFORMANCE
        )
        
        # Generate alert for critical errors
        if error_type in ['carbon_data_error', 'training_failure', 'system_error']:
            await self.generate_alert(
                AlertLevel.ERROR,
                "error_tracking",
                f"Error occurred: {error_type}",
                {"error_type": error_type, "details": details}
            )
    
    async def _check_thresholds(self) -> None:
        """Check metrics against defined thresholds."""
        for metric_name, thresholds in self.thresholds.items():
            recent_values = await self.get_recent_metric_values(metric_name, minutes=5)
            
            if not recent_values:
                continue
            
            # Use average of recent values
            avg_value = sum(recent_values) / len(recent_values)
            
            # Check thresholds
            if avg_value >= thresholds.get('critical', float('inf')):
                await self.generate_alert(
                    AlertLevel.CRITICAL,
                    "threshold_monitor",
                    f"{metric_name} critical: {avg_value:.2f}",
                    {"metric": metric_name, "value": avg_value, "threshold": thresholds['critical']}
                )
            elif avg_value >= thresholds.get('warning', float('inf')):
                await self.generate_alert(
                    AlertLevel.WARNING,
                    "threshold_monitor",
                    f"{metric_name} warning: {avg_value:.2f}",
                    {"metric": metric_name, "value": avg_value, "threshold": thresholds['warning']}
                )
    
    async def generate_alert(
        self,
        level: AlertLevel,
        component: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate system alert.
        
        Args:
            level: Alert severity level
            component: Component that generated the alert
            message: Alert message
            metadata: Additional alert metadata
            
        Returns:
            Alert ID
        """
        alert_id = f"{component}_{int(time.time() * 1000)}"
        
        # Check for similar recent alerts (cooldown)
        if await self._is_alert_in_cooldown(component, message):
            return alert_id
        
        alert = Alert(
            id=alert_id,
            level=level,
            component=component,
            message=message,
            metadata=metadata or {}
        )
        
        async with self._alert_lock:
            self._alerts.append(alert)
        
        # Notify alert callbacks
        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
        
        logger.log(
            logging.CRITICAL if level == AlertLevel.CRITICAL else
            logging.ERROR if level == AlertLevel.ERROR else
            logging.WARNING if level == AlertLevel.WARNING else
            logging.INFO,
            f"ALERT [{level.value.upper()}] {component}: {message}"
        )
        
        return alert_id
    
    async def _is_alert_in_cooldown(self, component: str, message: str) -> bool:
        """Check if similar alert is in cooldown period."""
        cutoff_time = datetime.now() - self.alert_cooldown
        
        async with self._alert_lock:
            for alert in reversed(self._alerts):  # Check recent alerts first
                if alert.timestamp < cutoff_time:
                    break
                
                if (alert.component == component and 
                    alert.message == message and 
                    not alert.resolved):
                    return True
        
        return False
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert by ID.
        
        Args:
            alert_id: Alert ID to resolve
            
        Returns:
            True if alert was found and resolved
        """
        async with self._alert_lock:
            for alert in self._alerts:
                if alert.id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolution_time = datetime.now()
                    logger.info(f"Alert resolved: {alert_id}")
                    return True
        
        return False
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert by ID.
        
        Args:
            alert_id: Alert ID to acknowledge
            
        Returns:
            True if alert was found and acknowledged
        """
        async with self._alert_lock:
            for alert in self._alerts:
                if alert.id == alert_id:
                    alert.acknowledged = True
                    logger.info(f"Alert acknowledged: {alert_id}")
                    return True
        
        return False
    
    async def get_recent_metric_values(
        self,
        metric_name: str,
        minutes: int = 60
    ) -> List[float]:
        """Get recent values for a metric.
        
        Args:
            metric_name: Name of the metric
            minutes: How many minutes of recent data to include
            
        Returns:
            List of recent metric values
        """
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        async with self._metric_lock:
            if metric_name not in self._metrics:
                return []
            
            return [
                m.value for m in self._metrics[metric_name]
                if m.timestamp >= cutoff_time
            ]
    
    async def get_active_alerts(self) -> List[Alert]:
        """Get all unresolved alerts.
        
        Returns:
            List of active (unresolved) alerts
        """
        async with self._alert_lock:
            return [alert for alert in self._alerts if not alert.resolved]
    
    async def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics.
        
        Returns:
            Alert summary including counts by level
        """
        active_alerts = await self.get_active_alerts()
        
        summary = {
            'total_active': len(active_alerts),
            'by_level': {},
            'by_component': {},
            'unacknowledged': 0
        }
        
        for alert in active_alerts:
            # Count by level
            level = alert.level.value
            summary['by_level'][level] = summary['by_level'].get(level, 0) + 1
            
            # Count by component
            component = alert.component
            summary['by_component'][component] = summary['by_component'].get(component, 0) + 1
            
            # Count unacknowledged
            if not alert.acknowledged:
                summary['unacknowledged'] += 1
        
        return summary
    
    async def get_metric_summary(self) -> Dict[str, Any]:
        """Get metric summary statistics.
        
        Returns:
            Metric summary including recent values and trends
        """
        summary = {
            'total_metrics': len(self._metrics),
            'recent_values': {},
            'system_uptime_hours': (datetime.now() - self._system_start_time).total_seconds() / 3600
        }
        
        # Get recent values for key metrics
        key_metrics = [
            'cpu_usage_percent', 'memory_usage_percent', 'disk_usage_percent',
            'carbon_intensity', 'training_loss'
        ]
        
        for metric in key_metrics:
            recent_values = await self.get_recent_metric_values(metric, minutes=5)
            if recent_values:
                summary['recent_values'][metric] = {
                    'current': recent_values[-1],
                    'average': sum(recent_values) / len(recent_values),
                    'min': min(recent_values),
                    'max': max(recent_values)
                }
        
        return summary
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old metrics and alerts."""
        now = datetime.now()
        
        # Only cleanup every hour
        if (now - self._last_cleanup).total_seconds() < 3600:
            return
        
        self._last_cleanup = now
        
        # Clean old metrics
        metric_cutoff = now - self.metric_retention
        async with self._metric_lock:
            for metric_name in list(self._metrics.keys()):
                self._metrics[metric_name] = [
                    m for m in self._metrics[metric_name]
                    if m.timestamp >= metric_cutoff
                ]
                
                # Remove empty metric lists
                if not self._metrics[metric_name]:
                    del self._metrics[metric_name]
        
        # Clean old alerts
        alert_cutoff = now - self.alert_retention
        async with self._alert_lock:
            self._alerts = [
                alert for alert in self._alerts
                if alert.timestamp >= alert_cutoff
            ]
        
        logger.debug("Completed data cleanup")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add callback for alert notifications.
        
        Args:
            callback: Function to call when alert is generated
        """
        self._alert_callbacks.append(callback)
    
    def set_threshold(self, metric_name: str, warning: float, critical: float) -> None:
        """Set custom threshold for a metric.
        
        Args:
            metric_name: Name of the metric
            warning: Warning threshold value
            critical: Critical threshold value
        """
        self.thresholds[metric_name] = {
            'warning': warning,
            'critical': critical
        }
        
        logger.info(f"Updated thresholds for {metric_name}: warning={warning}, critical={critical}")
    
    async def export_metrics_json(self) -> str:
        """Export all metrics as JSON.
        
        Returns:
            JSON string of all metrics
        """
        export_data = {
            'export_time': datetime.now().isoformat(),
            'system_uptime_hours': (datetime.now() - self._system_start_time).total_seconds() / 3600,
            'metrics': {}
        }
        
        async with self._metric_lock:
            for name, metrics in self._metrics.items():
                export_data['metrics'][name] = [
                    {
                        'value': m.value,
                        'unit': m.unit,
                        'type': m.metric_type.value,
                        'timestamp': m.timestamp.isoformat(),
                        'tags': m.tags
                    }
                    for m in metrics
                ]
        
        return json.dumps(export_data, indent=2)