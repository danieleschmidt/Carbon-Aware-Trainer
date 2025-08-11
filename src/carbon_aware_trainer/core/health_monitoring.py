"""
Comprehensive health monitoring and alerting system.
"""

import asyncio
import time
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

from .circuit_breaker import circuit_breaker_manager
from .exceptions import CarbonAwareException

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning" 
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Health check configuration."""
    name: str
    check_func: Callable[[], Any]
    timeout: float = 30.0
    interval: float = 60.0
    enabled: bool = True
    critical: bool = False
    last_check: Optional[datetime] = None
    last_status: HealthStatus = HealthStatus.UNKNOWN
    failure_count: int = 0
    max_failures: int = 3


@dataclass
class Alert:
    """System alert."""
    id: str
    severity: AlertSeverity
    component: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SystemMetrics:
    """System performance metrics."""
    
    def __init__(self):
        self.start_time = time.time()
        
    def get_cpu_metrics(self) -> Dict[str, Any]:
        """Get CPU metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            
            return {
                "cpu_percent": cpu_percent,
                "cpu_count": cpu_count,
                "load_average_1m": load_avg[0],
                "load_average_5m": load_avg[1],
                "load_average_15m": load_avg[2]
            }
        except Exception as e:
            logger.error(f"Failed to get CPU metrics: {e}")
            return {"error": str(e)}
            
    def get_memory_metrics(self) -> Dict[str, Any]:
        """Get memory metrics."""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                "memory_total_gb": memory.total / (1024**3),
                "memory_available_gb": memory.available / (1024**3),
                "memory_used_gb": memory.used / (1024**3),
                "memory_percent": memory.percent,
                "swap_total_gb": swap.total / (1024**3),
                "swap_used_gb": swap.used / (1024**3),
                "swap_percent": swap.percent
            }
        except Exception as e:
            logger.error(f"Failed to get memory metrics: {e}")
            return {"error": str(e)}
            
    def get_disk_metrics(self) -> Dict[str, Any]:
        """Get disk metrics."""
        try:
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            metrics = {
                "disk_total_gb": disk.total / (1024**3),
                "disk_used_gb": disk.used / (1024**3),
                "disk_free_gb": disk.free / (1024**3),
                "disk_percent": (disk.used / disk.total) * 100
            }
            
            if disk_io:
                metrics.update({
                    "disk_read_mb": disk_io.read_bytes / (1024**2),
                    "disk_write_mb": disk_io.write_bytes / (1024**2),
                    "disk_read_count": disk_io.read_count,
                    "disk_write_count": disk_io.write_count
                })
                
            return metrics
        except Exception as e:
            logger.error(f"Failed to get disk metrics: {e}")
            return {"error": str(e)}
            
    def get_network_metrics(self) -> Dict[str, Any]:
        """Get network metrics."""
        try:
            net_io = psutil.net_io_counters()
            
            if net_io:
                return {
                    "network_bytes_sent_mb": net_io.bytes_sent / (1024**2),
                    "network_bytes_recv_mb": net_io.bytes_recv / (1024**2),
                    "network_packets_sent": net_io.packets_sent,
                    "network_packets_recv": net_io.packets_recv,
                    "network_errors_in": net_io.errin,
                    "network_errors_out": net_io.errout,
                    "network_drops_in": net_io.dropin,
                    "network_drops_out": net_io.dropout
                }
            else:
                return {"error": "Network stats not available"}
        except Exception as e:
            logger.error(f"Failed to get network metrics: {e}")
            return {"error": str(e)}
            
    def get_process_metrics(self) -> Dict[str, Any]:
        """Get current process metrics."""
        try:
            process = psutil.Process()
            with process.oneshot():
                return {
                    "process_cpu_percent": process.cpu_percent(),
                    "process_memory_mb": process.memory_info().rss / (1024**2),
                    "process_memory_percent": process.memory_percent(),
                    "process_threads": process.num_threads(),
                    "process_fds": process.num_fds() if hasattr(process, 'num_fds') else 0,
                    "process_uptime_hours": (time.time() - self.start_time) / 3600
                }
        except Exception as e:
            logger.error(f"Failed to get process metrics: {e}")
            return {"error": str(e)}
            
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all system metrics."""
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu": self.get_cpu_metrics(),
            "memory": self.get_memory_metrics(),
            "disk": self.get_disk_metrics(),
            "network": self.get_network_metrics(),
            "process": self.get_process_metrics()
        }


class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.alerts: List[Alert] = []
        self.metrics = SystemMetrics()
        self.overall_status = HealthStatus.UNKNOWN
        self._monitoring_task: Optional[asyncio.Task] = None
        self._alert_handlers: List[Callable[[Alert], None]] = []
        
        # Register default health checks
        self._register_default_checks()
        
    def _register_default_checks(self):
        """Register default system health checks."""
        
        async def check_cpu_usage():
            metrics = self.metrics.get_cpu_metrics()
            if "error" in metrics:
                raise Exception(f"CPU check failed: {metrics['error']}")
            if metrics["cpu_percent"] > 90:
                raise Exception(f"High CPU usage: {metrics['cpu_percent']}%")
            return metrics["cpu_percent"]
            
        async def check_memory_usage():
            metrics = self.metrics.get_memory_metrics()
            if "error" in metrics:
                raise Exception(f"Memory check failed: {metrics['error']}")
            if metrics["memory_percent"] > 95:
                raise Exception(f"High memory usage: {metrics['memory_percent']}%")
            return metrics["memory_percent"]
            
        async def check_disk_space():
            metrics = self.metrics.get_disk_metrics() 
            if "error" in metrics:
                raise Exception(f"Disk check failed: {metrics['error']}")
            if metrics["disk_percent"] > 95:
                raise Exception(f"Low disk space: {metrics['disk_percent']}% used")
            return metrics["disk_percent"]
            
        async def check_circuit_breakers():
            health = circuit_breaker_manager.get_health_summary()
            if health["status"] == "degraded":
                raise Exception(f"Circuit breakers degraded: {health['open_breakers']} open")
            return health["status"]
            
        self.register_health_check("cpu_usage", check_cpu_usage, interval=60.0)
        self.register_health_check("memory_usage", check_memory_usage, interval=60.0)
        self.register_health_check("disk_space", check_disk_space, interval=300.0)
        self.register_health_check("circuit_breakers", check_circuit_breakers, interval=30.0, critical=True)
        
    def register_health_check(
        self,
        name: str,
        check_func: Callable,
        timeout: float = 30.0,
        interval: float = 60.0,
        critical: bool = False
    ):
        """Register a health check."""
        self.health_checks[name] = HealthCheck(
            name=name,
            check_func=check_func,
            timeout=timeout,
            interval=interval,
            critical=critical
        )
        logger.info(f"Registered health check: {name}")
        
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler function."""
        self._alert_handlers.append(handler)
        
    async def start_monitoring(self):
        """Start health monitoring."""
        if self._monitoring_task and not self._monitoring_task.done():
            logger.warning("Monitoring already running")
            return
            
        logger.info("Starting health monitoring")
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
    async def stop_monitoring(self):
        """Stop health monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped health monitoring")
        
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while True:
            try:
                # Run health checks
                await self._run_health_checks()
                
                # Update overall status
                self._update_overall_status()
                
                # Clean up old alerts
                self._cleanup_old_alerts()
                
                # Wait for next cycle
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)  # Back off on error
                
    async def _run_health_checks(self):
        """Run all enabled health checks."""
        current_time = datetime.now()
        
        for check in self.health_checks.values():
            if not check.enabled:
                continue
                
            # Check if it's time to run this check
            if (check.last_check and 
                (current_time - check.last_check).total_seconds() < check.interval):
                continue
                
            try:
                # Run health check with timeout
                result = await asyncio.wait_for(
                    check.check_func(),
                    timeout=check.timeout
                )
                
                # Check passed
                if check.last_status != HealthStatus.HEALTHY:
                    await self._create_alert(
                        f"{check.name}_recovered",
                        AlertSeverity.INFO,
                        check.name,
                        f"Health check {check.name} recovered"
                    )
                    
                check.last_status = HealthStatus.HEALTHY
                check.failure_count = 0
                
            except asyncio.TimeoutError:
                await self._handle_check_failure(check, "Health check timeout")
            except Exception as e:
                await self._handle_check_failure(check, str(e))
                
            check.last_check = current_time
            
    async def _handle_check_failure(self, check: HealthCheck, error_msg: str):
        """Handle health check failure."""
        check.failure_count += 1
        
        if check.failure_count >= check.max_failures:
            severity = AlertSeverity.CRITICAL if check.critical else AlertSeverity.ERROR
            check.last_status = HealthStatus.CRITICAL if check.critical else HealthStatus.WARNING
            
            await self._create_alert(
                f"{check.name}_failed",
                severity,
                check.name,
                f"Health check {check.name} failed: {error_msg}",
                {"failure_count": check.failure_count}
            )
        else:
            check.last_status = HealthStatus.WARNING
            
    async def _create_alert(
        self,
        alert_id: str,
        severity: AlertSeverity,
        component: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Create system alert."""
        alert = Alert(
            id=alert_id,
            severity=severity,
            component=component,
            message=message,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        logger.warning(f"Alert created: {severity.value.upper()} - {message}")
        
        # Notify alert handlers
        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
                
    def _update_overall_status(self):
        """Update overall system health status."""
        if not self.health_checks:
            self.overall_status = HealthStatus.UNKNOWN
            return
            
        statuses = [check.last_status for check in self.health_checks.values()]
        
        if any(s == HealthStatus.CRITICAL for s in statuses):
            self.overall_status = HealthStatus.CRITICAL
        elif any(s == HealthStatus.WARNING for s in statuses):
            self.overall_status = HealthStatus.WARNING
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            self.overall_status = HealthStatus.HEALTHY
        else:
            self.overall_status = HealthStatus.UNKNOWN
            
    def _cleanup_old_alerts(self):
        """Clean up old resolved alerts."""
        cutoff = datetime.now() - timedelta(hours=24)
        self.alerts = [
            alert for alert in self.alerts
            if not alert.resolved or alert.resolved_at > cutoff
        ]
        
    async def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        active_alerts = [a for a in self.alerts if not a.resolved]
        
        return {
            "overall_status": self.overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "health_checks": {
                name: {
                    "status": check.last_status.value,
                    "last_check": check.last_check.isoformat() if check.last_check else None,
                    "failure_count": check.failure_count,
                    "enabled": check.enabled
                }
                for name, check in self.health_checks.items()
            },
            "active_alerts": len(active_alerts),
            "total_alerts": len(self.alerts),
            "system_metrics": self.metrics.get_all_metrics()
        }
        
    def get_alerts(self, since_hours: int = 24) -> List[Dict[str, Any]]:
        """Get alerts from specified time period."""
        cutoff = datetime.now() - timedelta(hours=since_hours)
        recent_alerts = [a for a in self.alerts if a.timestamp > cutoff]
        
        return [
            {
                "id": alert.id,
                "severity": alert.severity.value,
                "component": alert.component,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "resolved": alert.resolved,
                "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
                "metadata": alert.metadata
            }
            for alert in recent_alerts
        ]
        
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert by ID."""
        for alert in self.alerts:
            if alert.id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                logger.info(f"Alert resolved: {alert_id}")
                return True
        return False
        
    def enable_check(self, name: str) -> bool:
        """Enable a health check."""
        if name in self.health_checks:
            self.health_checks[name].enabled = True
            logger.info(f"Enabled health check: {name}")
            return True
        return False
        
    def disable_check(self, name: str) -> bool:
        """Disable a health check."""
        if name in self.health_checks:
            self.health_checks[name].enabled = False
            logger.info(f"Disabled health check: {name}")
            return True
        return False


# Global health monitor instance
health_monitor = HealthMonitor()