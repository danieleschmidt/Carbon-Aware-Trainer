"""Production robustness enhancements for carbon-aware training."""

import asyncio
import logging
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum

from .types import CarbonIntensity, TrainingState, TrainingMetrics
from .exceptions import CarbonDataError, CarbonProviderError


logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DEGRADED = "degraded"


class RecoveryStrategy(str, Enum):
    """Recovery strategies for different failure modes."""
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    SWITCH_PROVIDER = "switch_provider"
    USE_FALLBACK_DATA = "use_fallback_data"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    EMERGENCY_PAUSE = "emergency_pause"


@dataclass
class HealthCheck:
    """Health check result."""
    component: str
    status: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, float] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)


@dataclass 
class RecoveryAction:
    """Automated recovery action."""
    strategy: RecoveryStrategy
    target_component: str
    executed_at: datetime
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)


class RobustnessManager:
    """Manages production robustness features for carbon-aware training."""
    
    def __init__(
        self,
        health_check_interval: int = 60,
        max_retry_attempts: int = 3,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: int = 300
    ):
        """Initialize robustness manager.
        
        Args:
            health_check_interval: Health check frequency in seconds
            max_retry_attempts: Maximum retry attempts for failed operations
            circuit_breaker_threshold: Failures before circuit breaker opens
            circuit_breaker_timeout: Circuit breaker reset timeout in seconds
        """
        self.health_check_interval = health_check_interval
        self.max_retry_attempts = max_retry_attempts
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        
        # Health monitoring
        self._health_checks: Dict[str, HealthCheck] = {}
        self._health_task: Optional[asyncio.Task] = None
        self._health_callbacks: List[Callable] = []
        
        # Circuit breaker state
        self._failure_counts: Dict[str, int] = {}
        self._circuit_breaker_open: Dict[str, datetime] = {}
        
        # Recovery tracking
        self._recovery_history: List[RecoveryAction] = []
        self._auto_recovery_enabled = True
        
        # Performance metrics
        self._start_time = datetime.now()
        self._uptime_seconds = 0
        self._total_operations = 0
        self._successful_operations = 0
        
        # Alerting
        self._alert_callbacks: List[Callable] = []
        self._last_alert_time: Dict[str, datetime] = {}
        self._alert_cooldown = 300  # 5 minutes
    
    async def start_health_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self._health_task and not self._health_task.done():
            return
        
        self._health_task = asyncio.create_task(self._health_monitoring_loop())
        logger.info("Started robustness health monitoring")
    
    async def stop_health_monitoring(self) -> None:
        """Stop health monitoring."""
        if self._health_task and not self._health_task.done():
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped robustness health monitoring")
    
    async def _health_monitoring_loop(self) -> None:
        """Main health monitoring loop."""
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _perform_health_checks(self) -> None:
        """Perform comprehensive health checks."""
        checks = []
        
        # System uptime and performance
        checks.append(self._check_system_performance())
        
        # Memory usage
        checks.append(self._check_memory_usage())
        
        # Error rates
        checks.append(self._check_error_rates())
        
        # Circuit breaker status
        checks.append(self._check_circuit_breakers())
        
        # Update health status
        for check in checks:
            self._health_checks[check.component] = check
            
            # Trigger alerts for critical issues
            if check.status == HealthStatus.CRITICAL:
                await self._trigger_alert(check)
            
            # Auto-recovery for certain issues
            if self._auto_recovery_enabled and check.status in [HealthStatus.CRITICAL, HealthStatus.DEGRADED]:
                await self._attempt_auto_recovery(check)
        
        # Notify health callbacks
        await self._notify_health_callbacks()
    
    def _check_system_performance(self) -> HealthCheck:
        """Check system performance metrics."""
        self._uptime_seconds = (datetime.now() - self._start_time).total_seconds()
        
        # Calculate success rate
        success_rate = 0.0
        if self._total_operations > 0:
            success_rate = self._successful_operations / self._total_operations
        
        status = HealthStatus.HEALTHY
        message = "System performing normally"
        suggestions = []
        
        if success_rate < 0.9:
            status = HealthStatus.WARNING
            message = f"Low success rate: {success_rate:.1%}"
            suggestions.append("Monitor for recurring errors")
        
        if success_rate < 0.7:
            status = HealthStatus.CRITICAL
            message = f"Critical success rate: {success_rate:.1%}"
            suggestions.extend([
                "Check data provider availability",
                "Verify network connectivity",
                "Consider switching to fallback mode"
            ])
        
        return HealthCheck(
            component="system_performance",
            status=status,
            message=message,
            metrics={
                "uptime_hours": self._uptime_seconds / 3600,
                "total_operations": self._total_operations,
                "success_rate": success_rate
            },
            suggestions=suggestions
        )
    
    def _check_memory_usage(self) -> HealthCheck:
        """Check memory usage patterns."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            status = HealthStatus.HEALTHY
            message = f"Memory usage: {memory_mb:.1f} MB"
            suggestions = []
            
            if memory_mb > 500:
                status = HealthStatus.WARNING
                message = f"High memory usage: {memory_mb:.1f} MB"
                suggestions.append("Monitor for memory leaks")
            
            if memory_mb > 1000:
                status = HealthStatus.CRITICAL
                message = f"Critical memory usage: {memory_mb:.1f} MB"
                suggestions.extend([
                    "Clear caches if possible",
                    "Restart application if necessary"
                ])
            
            return HealthCheck(
                component="memory_usage",
                status=status,
                message=message,
                metrics={"memory_mb": memory_mb},
                suggestions=suggestions
            )
        
        except ImportError:
            return HealthCheck(
                component="memory_usage",
                status=HealthStatus.WARNING,
                message="Memory monitoring unavailable (psutil not installed)",
                suggestions=["Install psutil for memory monitoring"]
            )
        except Exception as e:
            return HealthCheck(
                component="memory_usage", 
                status=HealthStatus.WARNING,
                message=f"Memory check failed: {e}",
                suggestions=["Investigate memory monitoring error"]
            )
    
    def _check_error_rates(self) -> HealthCheck:
        """Check recent error rates."""
        # Simple error rate based on failure counts
        total_failures = sum(self._failure_counts.values())
        
        status = HealthStatus.HEALTHY
        message = f"Total component failures: {total_failures}"
        suggestions = []
        
        if total_failures > 10:
            status = HealthStatus.WARNING
            message = f"Elevated failure count: {total_failures}"
            suggestions.append("Review recent error logs")
        
        if total_failures > 25:
            status = HealthStatus.CRITICAL
            message = f"High failure count: {total_failures}"
            suggestions.extend([
                "Investigate root cause of failures",
                "Consider switching to degraded mode"
            ])
        
        return HealthCheck(
            component="error_rates",
            status=status,
            message=message,
            metrics={"total_failures": total_failures},
            suggestions=suggestions
        )
    
    def _check_circuit_breakers(self) -> HealthCheck:
        """Check circuit breaker status."""
        open_breakers = len(self._circuit_breaker_open)
        
        status = HealthStatus.HEALTHY
        message = "All circuit breakers closed"
        
        if open_breakers > 0:
            status = HealthStatus.WARNING
            message = f"{open_breakers} circuit breakers open"
            
            # Check if any should be reset
            now = datetime.now()
            reset_candidates = []
            
            for component, open_time in list(self._circuit_breaker_open.items()):
                if (now - open_time).total_seconds() > self.circuit_breaker_timeout:
                    reset_candidates.append(component)
            
            if reset_candidates:
                message += f", {len(reset_candidates)} ready for reset"
        
        return HealthCheck(
            component="circuit_breakers",
            status=status,
            message=message,
            metrics={"open_breakers": open_breakers}
        )
    
    async def _trigger_alert(self, health_check: HealthCheck) -> None:
        """Trigger alert for critical health issues."""
        now = datetime.now()
        component = health_check.component
        
        # Rate limit alerts
        if component in self._last_alert_time:
            time_since_last = (now - self._last_alert_time[component]).total_seconds()
            if time_since_last < self._alert_cooldown:
                return
        
        self._last_alert_time[component] = now
        
        alert_data = {
            'component': component,
            'status': health_check.status,
            'message': health_check.message,
            'timestamp': now,
            'suggestions': health_check.suggestions
        }
        
        # Notify alert callbacks
        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback('health_alert', alert_data)
                else:
                    callback('health_alert', alert_data)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
        
        logger.warning(f"HEALTH ALERT - {component}: {health_check.message}")
    
    async def _attempt_auto_recovery(self, health_check: HealthCheck) -> None:
        """Attempt automated recovery for health issues."""
        component = health_check.component
        
        # Determine recovery strategy
        strategy = None
        if component == "system_performance" and health_check.status == HealthStatus.CRITICAL:
            strategy = RecoveryStrategy.GRACEFUL_DEGRADATION
        elif component == "memory_usage" and health_check.status == HealthStatus.CRITICAL:
            strategy = RecoveryStrategy.GRACEFUL_DEGRADATION
        elif component == "circuit_breakers":
            strategy = RecoveryStrategy.RETRY_WITH_BACKOFF
        
        if not strategy:
            return
        
        # Execute recovery action
        recovery_action = RecoveryAction(
            strategy=strategy,
            target_component=component,
            executed_at=datetime.now(),
            success=False
        )
        
        try:
            if strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                success = await self._execute_graceful_degradation(component)
            elif strategy == RecoveryStrategy.RETRY_WITH_BACKOFF:
                success = await self._execute_retry_with_backoff(component)
            else:
                success = False
            
            recovery_action.success = success
            
            if success:
                logger.info(f"Auto-recovery successful for {component} using {strategy.value}")
            else:
                logger.warning(f"Auto-recovery failed for {component} using {strategy.value}")
        
        except Exception as e:
            logger.error(f"Auto-recovery error for {component}: {e}")
            recovery_action.details['error'] = str(e)
        
        self._recovery_history.append(recovery_action)
    
    async def _execute_graceful_degradation(self, component: str) -> bool:
        """Execute graceful degradation strategy."""
        # Implementation would depend on specific component
        # For now, just log the action
        logger.info(f"Executing graceful degradation for {component}")
        return True
    
    async def _execute_retry_with_backoff(self, component: str) -> bool:
        """Execute retry with exponential backoff."""
        # Reset circuit breaker if timeout has passed
        now = datetime.now()
        if component in self._circuit_breaker_open:
            open_time = self._circuit_breaker_open[component]
            if (now - open_time).total_seconds() > self.circuit_breaker_timeout:
                del self._circuit_breaker_open[component]
                self._failure_counts[component] = 0
                logger.info(f"Circuit breaker reset for {component}")
                return True
        
        return False
    
    async def _notify_health_callbacks(self) -> None:
        """Notify health status callbacks."""
        health_summary = {
            'overall_status': self._calculate_overall_health(),
            'component_checks': {k: v for k, v in self._health_checks.items()},
            'timestamp': datetime.now()
        }
        
        for callback in self._health_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback('health_update', health_summary)
                else:
                    callback('health_update', health_summary)
            except Exception as e:
                logger.error(f"Health callback error: {e}")
    
    def _calculate_overall_health(self) -> HealthStatus:
        """Calculate overall system health status."""
        if not self._health_checks:
            return HealthStatus.WARNING
        
        statuses = [check.status for check in self._health_checks.values()]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def record_operation(self, success: bool) -> None:
        """Record operation result for health tracking."""
        self._total_operations += 1
        if success:
            self._successful_operations += 1
    
    def record_failure(self, component: str) -> None:
        """Record component failure for circuit breaker logic."""
        self._failure_counts[component] = self._failure_counts.get(component, 0) + 1
        
        # Check if circuit breaker should open
        if (self._failure_counts[component] >= self.circuit_breaker_threshold and 
            component not in self._circuit_breaker_open):
            self._circuit_breaker_open[component] = datetime.now()
            logger.warning(f"Circuit breaker opened for {component} after {self._failure_counts[component]} failures")
    
    def is_circuit_breaker_open(self, component: str) -> bool:
        """Check if circuit breaker is open for component."""
        return component in self._circuit_breaker_open
    
    def add_health_callback(self, callback: Callable) -> None:
        """Add health status change callback."""
        self._health_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable) -> None:
        """Add alert notification callback."""
        self._alert_callbacks.append(callback)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        return {
            'overall_status': self._calculate_overall_health().value,
            'uptime_hours': self._uptime_seconds / 3600,
            'total_operations': self._total_operations,
            'success_rate': (self._successful_operations / max(1, self._total_operations)),
            'component_health': {
                name: {
                    'status': check.status.value,
                    'message': check.message,
                    'metrics': check.metrics,
                    'last_check': check.timestamp.isoformat()
                }
                for name, check in self._health_checks.items()
            },
            'circuit_breakers': {
                'open_count': len(self._circuit_breaker_open),
                'open_components': list(self._circuit_breaker_open.keys())
            },
            'recent_recoveries': len([r for r in self._recovery_history if r.executed_at > datetime.now() - timedelta(hours=1)])
        }