"""Production reliability features for carbon-aware training."""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import contextlib

from .exceptions import CarbonAwareTrainerError, CarbonDataError, TrainingError
from .types import CarbonIntensity, TrainingState
from .enhanced_monitoring import EnhancedMonitoringSystem, AlertLevel

logger = logging.getLogger(__name__)


class ReliabilityLevel(str, Enum):
    """System reliability levels."""
    BASIC = "basic"
    ENHANCED = "enhanced"
    MISSION_CRITICAL = "mission_critical"


class FailoverMode(str, Enum):
    """Failover strategies."""
    IMMEDIATE = "immediate"
    GRACEFUL = "graceful"
    MAINTENANCE = "maintenance"


@dataclass
class CheckpointInfo:
    """Training checkpoint information."""
    step: int
    epoch: int
    timestamp: datetime
    file_path: str
    model_state_size: int
    optimizer_state_size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class ReliabilityMetrics:
    """Reliability metrics tracking."""
    uptime_seconds: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    mean_response_time_ms: float = 0.0
    error_rate_percent: float = 0.0
    last_checkpoint_time: Optional[datetime] = None
    recovery_count: int = 0
    failover_count: int = 0


class ProductionReliabilityManager:
    """Manages production reliability features."""
    
    def __init__(
        self,
        reliability_level: ReliabilityLevel = ReliabilityLevel.ENHANCED,
        checkpoint_interval_minutes: int = 15,
        health_check_interval_seconds: int = 30,
        backup_retention_days: int = 7,
        enable_circuit_breaker: bool = True,
        circuit_breaker_failure_threshold: int = 5,
        circuit_breaker_recovery_timeout: int = 300
    ):
        """Initialize production reliability manager.
        
        Args:
            reliability_level: Target reliability level
            checkpoint_interval_minutes: Auto-checkpoint frequency
            health_check_interval_seconds: Health check frequency
            backup_retention_days: How long to keep backup checkpoints
            enable_circuit_breaker: Enable circuit breaker pattern
            circuit_breaker_failure_threshold: Failures before circuit opens
            circuit_breaker_recovery_timeout: Seconds before circuit reset attempt
        """
        self.reliability_level = reliability_level
        self.checkpoint_interval = timedelta(minutes=checkpoint_interval_minutes)
        self.health_check_interval = timedelta(seconds=health_check_interval_seconds)
        self.backup_retention = timedelta(days=backup_retention_days)
        
        # Circuit breaker configuration
        self.enable_circuit_breaker = enable_circuit_breaker
        self.circuit_failure_threshold = circuit_breaker_failure_threshold
        self.circuit_recovery_timeout = circuit_breaker_recovery_timeout
        
        # State tracking
        self._start_time = datetime.now()
        self._metrics = ReliabilityMetrics()
        self._checkpoints: List[CheckpointInfo] = []
        self._is_healthy = True
        self._last_health_check = datetime.now()
        
        # Circuit breaker state
        self._circuit_breaker_open = {}
        self._circuit_failure_counts = {}
        self._circuit_last_failure = {}
        
        # Background tasks
        self._reliability_task: Optional[asyncio.Task] = None
        self._monitoring_system: Optional[EnhancedMonitoringSystem] = None
        
        # Callbacks
        self._health_callbacks: List[Callable] = []
        self._checkpoint_callbacks: List[Callable] = []
        self._recovery_callbacks: List[Callable] = []
        
        # Configuration based on reliability level
        self._configure_for_reliability_level()
        
        logger.info(f"Production reliability manager initialized at {reliability_level.value} level")
    
    def _configure_for_reliability_level(self) -> None:
        """Configure settings based on reliability level."""
        if self.reliability_level == ReliabilityLevel.MISSION_CRITICAL:
            # Most stringent settings
            self.checkpoint_interval = timedelta(minutes=5)
            self.health_check_interval = timedelta(seconds=10)
            self.backup_retention = timedelta(days=30)
            self.circuit_failure_threshold = 3
            self.circuit_recovery_timeout = 60
            
        elif self.reliability_level == ReliabilityLevel.ENHANCED:
            # Balanced settings
            self.checkpoint_interval = timedelta(minutes=15)
            self.health_check_interval = timedelta(seconds=30)
            self.backup_retention = timedelta(days=7)
            self.circuit_failure_threshold = 5
            self.circuit_recovery_timeout = 300
            
        else:  # BASIC
            # Relaxed settings for development
            self.checkpoint_interval = timedelta(minutes=30)
            self.health_check_interval = timedelta(seconds=60)
            self.backup_retention = timedelta(days=3)
            self.circuit_failure_threshold = 10
            self.circuit_recovery_timeout = 600
    
    async def start_reliability_monitoring(
        self,
        monitoring_system: Optional[EnhancedMonitoringSystem] = None
    ) -> None:
        """Start reliability monitoring and management.
        
        Args:
            monitoring_system: Optional monitoring system integration
        """
        if self._reliability_task and not self._reliability_task.done():
            return
        
        self._monitoring_system = monitoring_system
        self._reliability_task = asyncio.create_task(self._reliability_loop())
        
        logger.info("Started production reliability monitoring")
    
    async def stop_reliability_monitoring(self) -> None:
        """Stop reliability monitoring."""
        if self._reliability_task and not self._reliability_task.done():
            self._reliability_task.cancel()
            try:
                await self._reliability_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped production reliability monitoring")
    
    async def _reliability_loop(self) -> None:
        """Main reliability monitoring loop."""
        while True:
            try:
                # Perform health checks
                await self._perform_health_checks()
                
                # Check for checkpoint requirements
                await self._check_checkpoint_requirements()
                
                # Clean up old backups
                await self._cleanup_old_backups()
                
                # Update metrics
                await self._update_reliability_metrics()
                
                # Circuit breaker maintenance
                if self.enable_circuit_breaker:
                    await self._maintain_circuit_breakers()
                
                await asyncio.sleep(self.health_check_interval.total_seconds())
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Reliability monitoring error: {e}")
                await asyncio.sleep(self.health_check_interval.total_seconds())
    
    async def _perform_health_checks(self) -> None:
        """Perform comprehensive health checks."""
        health_issues = []
        
        try:
            # Check system resources
            resource_health = await self._check_system_resources()
            if resource_health:
                health_issues.extend(resource_health)
            
            # Check disk space
            disk_health = await self._check_disk_space()
            if disk_health:
                health_issues.extend(disk_health)
            
            # Check recent error rates
            error_health = await self._check_error_rates()
            if error_health:
                health_issues.extend(error_health)
            
            # Update health status
            was_healthy = self._is_healthy
            self._is_healthy = len(health_issues) == 0
            
            if was_healthy != self._is_healthy:
                await self._notify_health_change()
            
            self._last_health_check = datetime.now()
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            self._is_healthy = False
    
    async def _check_system_resources(self) -> List[str]:
        """Check system resource health."""
        issues = []
        
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                issues.append(f"High CPU usage: {cpu_percent}%")
            
            # Memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 85:
                issues.append(f"High memory usage: {memory.percent}%")
            
            # Load average (Linux/Unix)
            try:
                load_avg = psutil.getloadavg()[0]  # 1-minute load average
                cpu_count = psutil.cpu_count()
                if load_avg > cpu_count * 1.5:
                    issues.append(f"High load average: {load_avg}")
            except AttributeError:
                pass  # getloadavg not available on all platforms
            
        except ImportError:
            issues.append("System monitoring unavailable (psutil not installed)")
        except Exception as e:
            issues.append(f"System resource check failed: {e}")
        
        return issues
    
    async def _check_disk_space(self) -> List[str]:
        """Check disk space availability."""
        issues = []
        
        try:
            import shutil
            
            # Check current directory disk space
            total, used, free = shutil.disk_usage(".")
            free_percent = (free / total) * 100
            
            if free_percent < 10:
                issues.append(f"Low disk space: {free_percent:.1f}% free")
            elif free_percent < 20:
                issues.append(f"Disk space warning: {free_percent:.1f}% free")
            
        except Exception as e:
            issues.append(f"Disk space check failed: {e}")
        
        return issues
    
    async def _check_error_rates(self) -> List[str]:
        """Check recent error rates."""
        issues = []
        
        try:
            # Calculate error rate from recent requests
            total_recent = self._metrics.total_requests
            failed_recent = self._metrics.failed_requests
            
            if total_recent > 100:  # Only check if we have meaningful data
                error_rate = (failed_recent / total_recent) * 100
                
                if error_rate > 15:
                    issues.append(f"High error rate: {error_rate:.1f}%")
                elif error_rate > 5:
                    issues.append(f"Elevated error rate: {error_rate:.1f}%")
            
        except Exception as e:
            issues.append(f"Error rate check failed: {e}")
        
        return issues
    
    async def _check_checkpoint_requirements(self) -> None:
        """Check if automatic checkpointing is needed."""
        if not self._checkpoints:
            return  # No checkpoints to check against
        
        last_checkpoint = self._checkpoints[-1]
        time_since_checkpoint = datetime.now() - last_checkpoint.timestamp
        
        if time_since_checkpoint > self.checkpoint_interval:
            logger.info("Automatic checkpoint triggered by time interval")
            # Note: Actual checkpointing would be handled by training system
            await self._trigger_checkpoint_callbacks("time_interval")
    
    async def _cleanup_old_backups(self) -> None:
        """Clean up old backup checkpoints."""
        cutoff_time = datetime.now() - self.backup_retention
        
        # Find checkpoints to remove
        to_remove = []
        for i, checkpoint in enumerate(self._checkpoints):
            if checkpoint.timestamp < cutoff_time:
                to_remove.append(i)
        
        # Remove old checkpoints
        for i in reversed(to_remove):  # Remove from end to preserve indices
            old_checkpoint = self._checkpoints.pop(i)
            
            # Attempt to remove checkpoint file
            try:
                checkpoint_path = Path(old_checkpoint.file_path)
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    logger.debug(f"Removed old checkpoint: {old_checkpoint.file_path}")
            except Exception as e:
                logger.warning(f"Could not remove old checkpoint file {old_checkpoint.file_path}: {e}")
    
    async def _update_reliability_metrics(self) -> None:
        """Update reliability metrics."""
        self._metrics.uptime_seconds = (datetime.now() - self._start_time).total_seconds()
        
        # Calculate error rate
        if self._metrics.total_requests > 0:
            self._metrics.error_rate_percent = (
                self._metrics.failed_requests / self._metrics.total_requests
            ) * 100
        
        # Update last checkpoint time
        if self._checkpoints:
            self._metrics.last_checkpoint_time = self._checkpoints[-1].timestamp
        
        # Send metrics to monitoring system
        if self._monitoring_system:
            await self._monitoring_system.record_metric(
                "reliability_uptime_hours",
                self._metrics.uptime_seconds / 3600,
                "hours",
                "system"
            )
            
            await self._monitoring_system.record_metric(
                "reliability_error_rate",
                self._metrics.error_rate_percent,
                "percent",
                "system"
            )
    
    async def _maintain_circuit_breakers(self) -> None:
        """Maintain circuit breaker states."""
        current_time = datetime.now()
        
        # Check for circuits ready to close
        for component in list(self._circuit_breaker_open.keys()):
            open_time = self._circuit_breaker_open[component]
            
            if (current_time - open_time).total_seconds() > self.circuit_recovery_timeout:
                # Attempt to close circuit breaker
                await self._attempt_circuit_recovery(component)
    
    async def _attempt_circuit_recovery(self, component: str) -> None:
        """Attempt to recover a circuit breaker.
        
        Args:
            component: Component to recover
        """
        logger.info(f"Attempting circuit breaker recovery for {component}")
        
        try:
            # Test component health (this would be component-specific)
            # For now, just reset the circuit
            del self._circuit_breaker_open[component]
            self._circuit_failure_counts[component] = 0
            
            self._metrics.recovery_count += 1
            
            if self._monitoring_system:
                await self._monitoring_system.generate_alert(
                    AlertLevel.INFO,
                    "circuit_breaker",
                    f"Circuit breaker recovered for {component}",
                    {"component": component}
                )
            
            logger.info(f"Circuit breaker recovered for {component}")
            
        except Exception as e:
            logger.error(f"Circuit breaker recovery failed for {component}: {e}")
    
    async def record_request(self, success: bool, response_time_ms: float) -> None:
        """Record request outcome and timing.
        
        Args:
            success: Whether the request succeeded
            response_time_ms: Response time in milliseconds
        """
        self._metrics.total_requests += 1
        
        if success:
            self._metrics.successful_requests += 1
        else:
            self._metrics.failed_requests += 1
        
        # Update running average response time
        total_successful = self._metrics.successful_requests
        if total_successful > 0:
            self._metrics.mean_response_time_ms = (
                (self._metrics.mean_response_time_ms * (total_successful - 1) + response_time_ms) 
                / total_successful
            )
    
    async def record_component_failure(self, component: str, error: Exception) -> None:
        """Record component failure for circuit breaker logic.
        
        Args:
            component: Component that failed
            error: Exception that occurred
        """
        if not self.enable_circuit_breaker:
            return
        
        # Increment failure count
        self._circuit_failure_counts[component] = (
            self._circuit_failure_counts.get(component, 0) + 1
        )
        self._circuit_last_failure[component] = datetime.now()
        
        failure_count = self._circuit_failure_counts[component]
        
        # Open circuit breaker if threshold exceeded
        if (failure_count >= self.circuit_failure_threshold and 
            component not in self._circuit_breaker_open):
            
            self._circuit_breaker_open[component] = datetime.now()
            self._metrics.failover_count += 1
            
            if self._monitoring_system:
                await self._monitoring_system.generate_alert(
                    AlertLevel.WARNING,
                    "circuit_breaker",
                    f"Circuit breaker opened for {component} after {failure_count} failures",
                    {"component": component, "failure_count": failure_count, "error": str(error)}
                )
            
            logger.warning(f"Circuit breaker opened for {component} after {failure_count} failures")
    
    def is_circuit_breaker_open(self, component: str) -> bool:
        """Check if circuit breaker is open for component.
        
        Args:
            component: Component to check
            
        Returns:
            True if circuit breaker is open
        """
        return component in self._circuit_breaker_open
    
    async def create_checkpoint(
        self,
        step: int,
        epoch: int,
        file_path: str,
        model_state_size: int,
        optimizer_state_size: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Record a training checkpoint.
        
        Args:
            step: Training step
            epoch: Training epoch
            file_path: Path to checkpoint file
            model_state_size: Size of model state in bytes
            optimizer_state_size: Size of optimizer state in bytes
            metadata: Optional checkpoint metadata
            
        Returns:
            Checkpoint ID
        """
        checkpoint = CheckpointInfo(
            step=step,
            epoch=epoch,
            timestamp=datetime.now(),
            file_path=file_path,
            model_state_size=model_state_size,
            optimizer_state_size=optimizer_state_size,
            metadata=metadata or {}
        )
        
        self._checkpoints.append(checkpoint)
        
        # Keep only recent checkpoints in memory
        max_checkpoints = 50
        if len(self._checkpoints) > max_checkpoints:
            self._checkpoints = self._checkpoints[-max_checkpoints:]
        
        checkpoint_id = f"checkpoint_{step}_{int(checkpoint.timestamp.timestamp())}"
        
        if self._monitoring_system:
            await self._monitoring_system.record_metric(
                "checkpoint_size_mb",
                (model_state_size + optimizer_state_size) / (1024 * 1024),
                "MB",
                "training"
            )
        
        # Notify checkpoint callbacks
        await self._trigger_checkpoint_callbacks("manual", checkpoint)
        
        logger.info(f"Checkpoint created: {checkpoint_id} at step {step}")
        return checkpoint_id
    
    async def get_latest_checkpoint(self) -> Optional[CheckpointInfo]:
        """Get the most recent checkpoint.
        
        Returns:
            Latest checkpoint info or None
        """
        return self._checkpoints[-1] if self._checkpoints else None
    
    async def get_checkpoint_history(self, limit: int = 10) -> List[CheckpointInfo]:
        """Get recent checkpoint history.
        
        Args:
            limit: Maximum number of checkpoints to return
            
        Returns:
            List of recent checkpoints
        """
        return self._checkpoints[-limit:] if self._checkpoints else []
    
    def get_reliability_metrics(self) -> Dict[str, Any]:
        """Get comprehensive reliability metrics.
        
        Returns:
            Dictionary of reliability metrics
        """
        return {
            "reliability_level": self.reliability_level.value,
            "uptime_hours": self._metrics.uptime_seconds / 3600,
            "is_healthy": self._is_healthy,
            "last_health_check": self._last_health_check.isoformat(),
            "total_requests": self._metrics.total_requests,
            "success_rate_percent": (
                (self._metrics.successful_requests / max(1, self._metrics.total_requests)) * 100
            ),
            "error_rate_percent": self._metrics.error_rate_percent,
            "mean_response_time_ms": self._metrics.mean_response_time_ms,
            "total_checkpoints": len(self._checkpoints),
            "last_checkpoint": (
                self._metrics.last_checkpoint_time.isoformat()
                if self._metrics.last_checkpoint_time else None
            ),
            "recovery_count": self._metrics.recovery_count,
            "failover_count": self._metrics.failover_count,
            "circuit_breakers": {
                "open_count": len(self._circuit_breaker_open),
                "open_components": list(self._circuit_breaker_open.keys())
            }
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status.
        
        Returns:
            Health status information
        """
        return {
            "is_healthy": self._is_healthy,
            "last_check": self._last_health_check.isoformat(),
            "uptime_hours": (datetime.now() - self._start_time).total_seconds() / 3600,
            "reliability_level": self.reliability_level.value
        }
    
    async def _notify_health_change(self) -> None:
        """Notify callbacks about health status change."""
        health_data = {
            "is_healthy": self._is_healthy,
            "timestamp": datetime.now(),
            "metrics": self.get_reliability_metrics()
        }
        
        for callback in self._health_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback("health_change", health_data)
                else:
                    callback("health_change", health_data)
            except Exception as e:
                logger.error(f"Health callback error: {e}")
    
    async def _trigger_checkpoint_callbacks(
        self, 
        trigger_type: str, 
        checkpoint: Optional[CheckpointInfo] = None
    ) -> None:
        """Trigger checkpoint-related callbacks.
        
        Args:
            trigger_type: Type of checkpoint trigger
            checkpoint: Checkpoint info (if available)
        """
        callback_data = {
            "trigger_type": trigger_type,
            "timestamp": datetime.now(),
            "checkpoint": checkpoint
        }
        
        for callback in self._checkpoint_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback("checkpoint_required", callback_data)
                else:
                    callback("checkpoint_required", callback_data)
            except Exception as e:
                logger.error(f"Checkpoint callback error: {e}")
    
    def add_health_callback(self, callback: Callable) -> None:
        """Add health status change callback.
        
        Args:
            callback: Function to call on health changes
        """
        self._health_callbacks.append(callback)
    
    def add_checkpoint_callback(self, callback: Callable) -> None:
        """Add checkpoint requirement callback.
        
        Args:
            callback: Function to call when checkpoint is needed
        """
        self._checkpoint_callbacks.append(callback)
    
    def add_recovery_callback(self, callback: Callable) -> None:
        """Add recovery event callback.
        
        Args:
            callback: Function to call on recovery events
        """
        self._recovery_callbacks.append(callback)
    
    @contextlib.asynccontextmanager
    async def request_context(self):
        """Context manager for tracking request reliability."""
        start_time = time.time()
        success = False
        
        try:
            yield
            success = True
        except Exception as e:
            success = False
            raise
        finally:
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            await self.record_request(success, response_time)
    
    async def perform_graceful_shutdown(self, timeout_seconds: int = 30) -> None:
        """Perform graceful shutdown of reliability monitoring.
        
        Args:
            timeout_seconds: Maximum time to wait for shutdown
        """
        logger.info("Starting graceful reliability monitoring shutdown")
        
        # Stop monitoring task
        await self.stop_reliability_monitoring()
        
        # Save final metrics
        final_metrics = self.get_reliability_metrics()
        logger.info(f"Final reliability metrics: {json.dumps(final_metrics, indent=2)}")
        
        # Notify callbacks of shutdown
        for callback in self._recovery_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback("graceful_shutdown", final_metrics)
                else:
                    callback("graceful_shutdown", final_metrics)
            except Exception as e:
                logger.error(f"Shutdown callback error: {e}")
        
        logger.info("Graceful reliability monitoring shutdown complete")