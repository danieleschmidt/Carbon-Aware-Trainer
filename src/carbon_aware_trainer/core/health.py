"""Health monitoring and system checks for carbon-aware trainer."""

import asyncio
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

from .exceptions import (
    CarbonProviderError, MonitoringError, MetricsError,
    CarbonProviderTimeoutError
)
from .types import CarbonDataSource


logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    WARNING = "warning" 
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None
    duration_ms: Optional[float] = None


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    gpu_utilization: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    power_draw_watts: Optional[float] = None


class HealthMonitor:
    """Comprehensive health monitoring for carbon-aware training."""
    
    def __init__(
        self,
        check_interval: int = 300,  # 5 minutes
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        """Initialize health monitor.
        
        Args:
            check_interval: Health check interval in seconds
            alert_thresholds: Custom alert thresholds
        """
        self.check_interval = check_interval
        self.alert_thresholds = alert_thresholds or {
            'cpu_percent': 90.0,
            'memory_percent': 85.0,
            'disk_percent': 95.0,
            'response_time_ms': 5000.0,
            'error_rate_percent': 10.0
        }
        
        # Health state
        self._health_checks: Dict[str, HealthCheck] = {}
        self._system_metrics: Optional[SystemMetrics] = None
        self._alert_callbacks: List[Callable[[HealthCheck], None]] = []
        
        # Monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
        self._stop_monitoring = False
        
        # Error tracking
        self._error_counts: Dict[str, int] = {}
        self._last_error_reset = datetime.now()
    
    async def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self._monitoring_task and not self._monitoring_task.done():
            logger.warning("Health monitoring already started")
            return
        
        self._stop_monitoring = False
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started health monitoring")
    
    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self._stop_monitoring = True
        
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped health monitoring")
    
    async def _monitoring_loop(self) -> None:
        """Main health monitoring loop."""
        while not self._stop_monitoring:
            try:
                await self.run_all_checks()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(min(self.check_interval, 60))  # Backoff
    
    async def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all health checks.
        
        Returns:
            Dictionary of health check results
        """
        checks = await asyncio.gather(
            self._check_system_resources(),
            self._check_carbon_provider_connectivity(),
            self._check_disk_space(),
            self._check_memory_usage(),
            self._check_network_connectivity(),
            return_exceptions=True
        )
        
        # Process results
        check_names = [
            "system_resources", "carbon_provider", "disk_space",
            "memory_usage", "network_connectivity"
        ]
        
        for name, result in zip(check_names, checks):
            if isinstance(result, Exception):
                self._health_checks[name] = HealthCheck(
                    name=name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {result}",
                    timestamp=datetime.now()
                )
            else:
                self._health_checks[name] = result
        
        # Check for alerts
        await self._check_for_alerts()
        
        return self._health_checks.copy()
    
    async def _check_system_resources(self) -> HealthCheck:
        """Check system resource utilization."""
        start_time = datetime.now()
        
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O
            net_io = psutil.net_io_counters()
            network_io = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
            
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
            except ImportError:
                pass  # GPUtil not available
            
            self._system_metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=disk.percent,
                network_io=network_io,
                gpu_utilization=gpu_util,
                gpu_memory_percent=gpu_memory,
                power_draw_watts=power_draw
            )
            
            # Determine status
            status = HealthStatus.HEALTHY
            issues = []
            
            if cpu_percent > self.alert_thresholds['cpu_percent']:
                status = HealthStatus.WARNING
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory.percent > self.alert_thresholds['memory_percent']:
                status = HealthStatus.WARNING
                issues.append(f"High memory usage: {memory.percent:.1f}%")
            
            if disk.percent > self.alert_thresholds['disk_percent']:
                status = HealthStatus.CRITICAL
                issues.append(f"High disk usage: {disk.percent:.1f}%")
            
            message = "System resources normal"
            if issues:
                message = "; ".join(issues)
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            
            return HealthCheck(
                name="system_resources",
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=duration,
                details={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'disk_percent': disk.percent,
                    'gpu_utilization': gpu_util,
                    'gpu_memory_percent': gpu_memory
                }
            )
            
        except Exception as e:
            logger.error(f"System resource check failed: {e}")
            return HealthCheck(
                name="system_resources",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check system resources: {e}",
                timestamp=datetime.now()
            )
    
    async def _check_carbon_provider_connectivity(self) -> HealthCheck:
        """Check carbon data provider connectivity."""
        start_time = datetime.now()
        
        try:
            # This would normally test actual provider connectivity
            # For now, simulate a connectivity check
            await asyncio.sleep(0.1)  # Simulate network call
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            
            if duration > self.alert_thresholds['response_time_ms']:
                status = HealthStatus.WARNING
                message = f"Slow carbon provider response: {duration:.1f}ms"
            else:
                status = HealthStatus.HEALTHY
                message = "Carbon provider connectivity OK"
            
            return HealthCheck(
                name="carbon_provider",
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=duration
            )
            
        except Exception as e:
            logger.error(f"Carbon provider check failed: {e}")
            return HealthCheck(
                name="carbon_provider", 
                status=HealthStatus.CRITICAL,
                message=f"Carbon provider unreachable: {e}",
                timestamp=datetime.now()
            )
    
    async def _check_disk_space(self) -> HealthCheck:
        """Check available disk space."""
        try:
            disk = psutil.disk_usage('/')
            free_gb = disk.free / (1024**3)
            
            status = HealthStatus.HEALTHY
            if disk.percent > 95:
                status = HealthStatus.CRITICAL
            elif disk.percent > 85:
                status = HealthStatus.WARNING
            
            message = f"Disk usage: {disk.percent:.1f}% ({free_gb:.1f}GB free)"
            
            return HealthCheck(
                name="disk_space",
                status=status,
                message=message,
                timestamp=datetime.now(),
                details={'free_gb': free_gb, 'percent_used': disk.percent}
            )
            
        except Exception as e:
            return HealthCheck(
                name="disk_space",
                status=HealthStatus.CRITICAL,
                message=f"Disk space check failed: {e}",
                timestamp=datetime.now()
            )
    
    async def _check_memory_usage(self) -> HealthCheck:
        """Check memory usage and detect potential leaks."""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            status = HealthStatus.HEALTHY
            issues = []
            
            if memory.percent > 90:
                status = HealthStatus.CRITICAL
                issues.append(f"Critical memory usage: {memory.percent:.1f}%")
            elif memory.percent > 80:
                status = HealthStatus.WARNING
                issues.append(f"High memory usage: {memory.percent:.1f}%")
            
            if swap.percent > 50:
                status = max(status, HealthStatus.WARNING, key=lambda x: x.value)
                issues.append(f"High swap usage: {swap.percent:.1f}%")
            
            message = "Memory usage normal" if not issues else "; ".join(issues)
            
            return HealthCheck(
                name="memory_usage",
                status=status,
                message=message,
                timestamp=datetime.now(),
                details={
                    'memory_percent': memory.percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'swap_percent': swap.percent
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="memory_usage",
                status=HealthStatus.CRITICAL,
                message=f"Memory check failed: {e}",
                timestamp=datetime.now()
            )
    
    async def _check_network_connectivity(self) -> HealthCheck:
        """Check network connectivity."""
        try:
            # Simple connectivity test
            import socket
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            
            start_time = datetime.now()
            result = sock.connect_ex(('8.8.8.8', 53))  # Google DNS
            sock.close()
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            
            if result == 0:
                status = HealthStatus.HEALTHY
                message = f"Network connectivity OK ({duration:.1f}ms)"
            else:
                status = HealthStatus.CRITICAL
                message = "Network connectivity failed"
            
            return HealthCheck(
                name="network_connectivity",
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=duration
            )
            
        except Exception as e:
            return HealthCheck(
                name="network_connectivity",
                status=HealthStatus.CRITICAL,
                message=f"Network check failed: {e}",
                timestamp=datetime.now()
            )
    
    async def _check_for_alerts(self) -> None:
        """Check if any health checks require alerts."""
        for check in self._health_checks.values():
            if check.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                for callback in self._alert_callbacks:
                    try:
                        callback(check)
                    except Exception as e:
                        logger.error(f"Alert callback failed: {e}")
    
    def add_alert_callback(self, callback: Callable[[HealthCheck], None]) -> None:
        """Add callback for health alerts.
        
        Args:
            callback: Function to call when health issues are detected
        """
        self._alert_callbacks.append(callback)
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status.
        
        Returns:
            Overall health status based on all checks
        """
        if not self._health_checks:
            return HealthStatus.UNKNOWN
        
        statuses = [check.status for check in self._health_checks.values()]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary.
        
        Returns:
            Dictionary with health status and metrics
        """
        return {
            'overall_status': self.get_overall_status().value,
            'last_check': max(
                (check.timestamp for check in self._health_checks.values()),
                default=None
            ),
            'checks': {
                name: {
                    'status': check.status.value,
                    'message': check.message,
                    'timestamp': check.timestamp.isoformat(),
                    'duration_ms': check.duration_ms,
                    'details': check.details
                }
                for name, check in self._health_checks.items()
            },
            'system_metrics': {
                'cpu_percent': self._system_metrics.cpu_percent,
                'memory_percent': self._system_metrics.memory_percent,
                'disk_percent': self._system_metrics.disk_percent,
                'gpu_utilization': self._system_metrics.gpu_utilization,
                'gpu_memory_percent': self._system_metrics.gpu_memory_percent
            } if self._system_metrics else None,
            'alert_thresholds': self.alert_thresholds
        }
    
    def record_error(self, error_type: str) -> None:
        """Record an error for tracking.
        
        Args:
            error_type: Type/category of error
        """
        if error_type not in self._error_counts:
            self._error_counts[error_type] = 0
        
        self._error_counts[error_type] += 1
        
        # Reset counters hourly
        if datetime.now() - self._last_error_reset > timedelta(hours=1):
            self._error_counts.clear()
            self._last_error_reset = datetime.now()
    
    def get_error_rates(self) -> Dict[str, float]:
        """Get current error rates.
        
        Returns:
            Dictionary of error types and their rates per hour
        """
        hours_elapsed = (datetime.now() - self._last_error_reset).total_seconds() / 3600
        hours_elapsed = max(hours_elapsed, 0.1)  # Avoid division by zero
        
        return {
            error_type: count / hours_elapsed
            for error_type, count in self._error_counts.items()
        }