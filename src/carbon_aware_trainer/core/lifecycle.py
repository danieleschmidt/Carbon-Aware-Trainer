"""Production-grade application lifecycle management with graceful shutdown."""

import asyncio
import signal
import sys
import logging
from typing import List, Callable, Optional, Dict, Any
from datetime import datetime
import threading
from contextlib import asynccontextmanager

from .exceptions import CarbonAwareTrainerError
from .alerting import alerting_manager, Alert, AlertType, AlertSeverity
from .metrics_collector import metrics_collector
from .health import HealthMonitor
from .config import config_manager


logger = logging.getLogger(__name__)


class GracefulShutdownHandler:
    """Handles graceful shutdown with proper resource cleanup."""
    
    def __init__(self, shutdown_timeout: int = 30):
        """Initialize shutdown handler.
        
        Args:
            shutdown_timeout: Maximum time to wait for graceful shutdown
        """
        self.shutdown_timeout = shutdown_timeout
        self.shutdown_callbacks: List[Callable] = []
        self.is_shutting_down = False
        self.shutdown_event = asyncio.Event()
        
        # Register signal handlers
        self._setup_signal_handlers()
        
        logger.info("GracefulShutdownHandler initialized")
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())
        
        # Register handlers for common shutdown signals
        for sig in [signal.SIGTERM, signal.SIGINT]:
            if hasattr(signal, sig.name):
                signal.signal(sig, signal_handler)
    
    def add_shutdown_callback(self, callback: Callable) -> None:
        """Add callback to be executed during shutdown.
        
        Args:
            callback: Function to call during shutdown
        """
        self.shutdown_callbacks.append(callback)
        logger.debug(f"Added shutdown callback: {callback.__name__}")
    
    def remove_shutdown_callback(self, callback: Callable) -> None:
        """Remove shutdown callback.
        
        Args:
            callback: Callback to remove
        """
        if callback in self.shutdown_callbacks:
            self.shutdown_callbacks.remove(callback)
            logger.debug(f"Removed shutdown callback: {callback.__name__}")
    
    async def shutdown(self) -> None:
        """Execute graceful shutdown sequence."""
        if self.is_shutting_down:
            logger.warning("Shutdown already in progress")
            return
        
        self.is_shutting_down = True
        shutdown_start = datetime.now()
        
        logger.info("Starting graceful shutdown sequence...")
        
        try:
            # Create alert for shutdown
            await alerting_manager.create_alert(
                alert_type=AlertType.SYSTEM_HEALTH,
                severity=AlertSeverity.INFO,
                title="System shutdown initiated",
                message="Graceful shutdown sequence started",
                source="lifecycle_manager"
            )
            
            # Execute shutdown callbacks
            await self._execute_shutdown_callbacks()
            
            # Stop core services
            await self._stop_core_services()
            
            # Final cleanup
            await self._final_cleanup()
            
            shutdown_duration = (datetime.now() - shutdown_start).total_seconds()
            logger.info(f"Graceful shutdown completed in {shutdown_duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            self.shutdown_event.set()
    
    async def _execute_shutdown_callbacks(self) -> None:
        """Execute all registered shutdown callbacks."""
        logger.info(f"Executing {len(self.shutdown_callbacks)} shutdown callbacks...")
        
        for callback in self.shutdown_callbacks:
            try:
                callback_start = datetime.now()
                
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
                
                callback_duration = (datetime.now() - callback_start).total_seconds()
                logger.debug(f"Shutdown callback {callback.__name__} completed in {callback_duration:.2f}s")
                
            except Exception as e:
                logger.error(f"Error in shutdown callback {callback.__name__}: {e}")
    
    async def _stop_core_services(self) -> None:
        """Stop core system services."""
        logger.info("Stopping core services...")
        
        services_to_stop = [
            ("MetricsCollector", metrics_collector.stop),
            ("AlertingManager", alerting_manager.stop),
        ]
        
        # Stop services concurrently
        stop_tasks = []
        for service_name, stop_method in services_to_stop:
            if asyncio.iscoroutinefunction(stop_method):
                stop_tasks.append(self._stop_service_with_timeout(service_name, stop_method))
            else:
                try:
                    stop_method()
                    logger.debug(f"Stopped {service_name}")
                except Exception as e:
                    logger.error(f"Error stopping {service_name}: {e}")
        
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
    
    async def _stop_service_with_timeout(self, service_name: str, stop_method: Callable) -> None:
        """Stop a service with timeout protection.
        
        Args:
            service_name: Name of service for logging
            stop_method: Async stop method to call
        """
        try:
            await asyncio.wait_for(stop_method(), timeout=10.0)
            logger.debug(f"Stopped {service_name}")
        except asyncio.TimeoutError:
            logger.warning(f"Timeout stopping {service_name}")
        except Exception as e:
            logger.error(f"Error stopping {service_name}: {e}")
    
    async def _final_cleanup(self) -> None:
        """Perform final cleanup tasks."""
        logger.info("Performing final cleanup...")
        
        # Cancel any remaining tasks
        tasks = [task for task in asyncio.all_tasks() if not task.done()]
        if tasks:
            logger.info(f"Cancelling {len(tasks)} remaining tasks...")
            for task in tasks:
                task.cancel()
            
            # Wait for tasks to complete cancellation
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown to complete."""
        await self.shutdown_event.wait()


class HealthEndpoint:
    """HTTP health check endpoint for monitoring systems."""
    
    def __init__(self, health_monitor: HealthMonitor, port: int = 8080):
        """Initialize health endpoint.
        
        Args:
            health_monitor: Health monitor instance
            port: Port to serve health endpoint
        """
        self.health_monitor = health_monitor
        self.port = port
        self.server: Optional[Any] = None
        
        logger.info(f"HealthEndpoint initialized on port {port}")
    
    async def start(self) -> None:
        """Start health endpoint server."""
        try:
            from aiohttp import web, web_runner
            
            app = web.Application()
            app.router.add_get('/health', self._health_handler)
            app.router.add_get('/health/detailed', self._detailed_health_handler)
            app.router.add_get('/metrics', self._metrics_handler)
            app.router.add_get('/ready', self._readiness_handler)
            
            runner = web_runner.AppRunner(app)
            await runner.setup()
            
            site = web_runner.TCPSite(runner, '0.0.0.0', self.port)
            await site.start()
            
            self.server = runner
            logger.info(f"Health endpoint started on port {self.port}")
            
        except ImportError:
            logger.warning("aiohttp not available, health endpoint disabled")
        except Exception as e:
            logger.error(f"Failed to start health endpoint: {e}")
    
    async def stop(self) -> None:
        """Stop health endpoint server."""
        if self.server:
            try:
                await self.server.cleanup()
                logger.info("Health endpoint stopped")
            except Exception as e:
                logger.error(f"Error stopping health endpoint: {e}")
    
    async def _health_handler(self, request) -> Any:
        """Handle basic health check."""
        from aiohttp import web
        
        try:
            overall_status = self.health_monitor.get_overall_status()
            
            if overall_status.value in ['healthy', 'warning']:
                return web.json_response({
                    'status': overall_status.value,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return web.json_response({
                    'status': overall_status.value,
                    'timestamp': datetime.now().isoformat()
                }, status=503)
                
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return web.json_response({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, status=500)
    
    async def _detailed_health_handler(self, request) -> Any:
        """Handle detailed health check."""
        from aiohttp import web
        
        try:
            health_summary = self.health_monitor.get_health_summary()
            
            status_code = 200
            if health_summary['overall_status'] == 'critical':
                status_code = 503
            elif health_summary['overall_status'] == 'warning':
                status_code = 200  # Warning is still OK for detailed view
            
            return web.json_response(health_summary, status=status_code)
            
        except Exception as e:
            logger.error(f"Detailed health check error: {e}")
            return web.json_response({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, status=500)
    
    async def _metrics_handler(self, request) -> Any:
        """Handle metrics endpoint."""
        from aiohttp import web
        
        try:
            # Export metrics in Prometheus format
            metrics_data = metrics_collector.export_metrics(format='prometheus')
            return web.Response(text=metrics_data, content_type='text/plain')
            
        except Exception as e:
            logger.error(f"Metrics endpoint error: {e}")
            return web.Response(text=f"# Error: {e}", status=500)
    
    async def _readiness_handler(self, request) -> Any:
        """Handle readiness check."""
        from aiohttp import web
        
        try:
            # Check if all critical services are ready
            config = config_manager.get_config()
            
            readiness_checks = {
                'config_loaded': config is not None,
                'metrics_collector_running': hasattr(metrics_collector, '_running') and metrics_collector._running,
                'alerting_manager_running': hasattr(alerting_manager, '_running') and alerting_manager._running,
            }
            
            all_ready = all(readiness_checks.values())
            
            response_data = {
                'ready': all_ready,
                'checks': readiness_checks,
                'timestamp': datetime.now().isoformat()
            }
            
            status_code = 200 if all_ready else 503
            return web.json_response(response_data, status=status_code)
            
        except Exception as e:
            logger.error(f"Readiness check error: {e}")
            return web.json_response({
                'ready': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, status=500)


class ApplicationLifecycleManager:
    """Manages the complete application lifecycle."""
    
    def __init__(self):
        """Initialize lifecycle manager."""
        self.shutdown_handler = GracefulShutdownHandler()
        self.health_monitor = HealthMonitor()
        self.health_endpoint: Optional[HealthEndpoint] = None
        
        self._startup_tasks: List[Callable] = []
        self._shutdown_tasks: List[Callable] = []
        self._running = False
        
        logger.info("ApplicationLifecycleManager initialized")
    
    def add_startup_task(self, task: Callable) -> None:
        """Add task to be executed during startup.
        
        Args:
            task: Startup task function
        """
        self._startup_tasks.append(task)
        logger.debug(f"Added startup task: {task.__name__}")
    
    def add_shutdown_task(self, task: Callable) -> None:
        """Add task to be executed during shutdown.
        
        Args:
            task: Shutdown task function
        """
        self._shutdown_tasks.append(task)
        self.shutdown_handler.add_shutdown_callback(task)
        logger.debug(f"Added shutdown task: {task.__name__}")
    
    @asynccontextmanager
    async def lifespan(self):
        """Async context manager for application lifespan."""
        try:
            await self.startup()
            yield
        finally:
            await self.shutdown()
    
    async def startup(self) -> None:
        """Execute application startup sequence."""
        if self._running:
            logger.warning("Application already running")
            return
        
        logger.info("Starting application...")
        startup_start = datetime.now()
        
        try:
            # Load configuration
            await self._load_configuration()
            
            # Execute startup tasks
            await self._execute_startup_tasks()
            
            # Start core services
            await self._start_core_services()
            
            # Start health endpoint
            await self._start_health_endpoint()
            
            self._running = True
            startup_duration = (datetime.now() - startup_start).total_seconds()
            
            logger.info(f"Application started successfully in {startup_duration:.2f}s")
            
            # Create startup alert
            await alerting_manager.create_alert(
                alert_type=AlertType.SYSTEM_HEALTH,
                severity=AlertSeverity.INFO,
                title="Application started",
                message=f"Carbon Aware Trainer started successfully in {startup_duration:.2f}s",
                source="lifecycle_manager"
            )
            
        except Exception as e:
            logger.error(f"Failed to start application: {e}")
            await self.shutdown()
            raise
    
    async def shutdown(self) -> None:
        """Execute application shutdown sequence."""
        if not self._running:
            logger.debug("Application not running, skipping shutdown")
            return
        
        logger.info("Shutting down application...")
        self._running = False
        
        try:
            # Stop health endpoint
            if self.health_endpoint:
                await self.health_endpoint.stop()
            
            # Execute graceful shutdown
            await self.shutdown_handler.shutdown()
            
            logger.info("Application shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during application shutdown: {e}")
    
    async def _load_configuration(self) -> None:
        """Load application configuration."""
        try:
            config_manager.load_config()
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    async def _execute_startup_tasks(self) -> None:
        """Execute all startup tasks."""
        logger.info(f"Executing {len(self._startup_tasks)} startup tasks...")
        
        for task in self._startup_tasks:
            try:
                task_start = datetime.now()
                
                if asyncio.iscoroutinefunction(task):
                    await task()
                else:
                    task()
                
                task_duration = (datetime.now() - task_start).total_seconds()
                logger.debug(f"Startup task {task.__name__} completed in {task_duration:.2f}s")
                
            except Exception as e:
                logger.error(f"Startup task {task.__name__} failed: {e}")
                raise
    
    async def _start_core_services(self) -> None:
        """Start core application services."""
        logger.info("Starting core services...")
        
        try:
            # Start services in order
            await metrics_collector.start()
            await alerting_manager.start()
            await self.health_monitor.start_monitoring()
            
            logger.info("Core services started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start core services: {e}")
            raise
    
    async def _start_health_endpoint(self) -> None:
        """Start health check endpoint."""
        try:
            config = config_manager.get_config()
            
            # Check if health endpoint is enabled in config
            health_port = getattr(config, 'health_endpoint_port', 8080)
            
            self.health_endpoint = HealthEndpoint(self.health_monitor, health_port)
            await self.health_endpoint.start()
            
        except Exception as e:
            logger.warning(f"Health endpoint not started: {e}")
    
    def is_running(self) -> bool:
        """Check if application is running.
        
        Returns:
            True if application is running
        """
        return self._running
    
    def get_status(self) -> Dict[str, Any]:
        """Get application status.
        
        Returns:
            Dictionary with status information
        """
        return {
            'running': self._running,
            'startup_tasks_count': len(self._startup_tasks),
            'shutdown_tasks_count': len(self._shutdown_tasks),
            'shutdown_in_progress': self.shutdown_handler.is_shutting_down,
            'health_endpoint_enabled': self.health_endpoint is not None
        }


# Global lifecycle manager instance
lifecycle_manager = ApplicationLifecycleManager()


def setup_production_lifecycle():
    """Setup production-ready application lifecycle with all components."""
    
    # Add startup tasks
    async def initialize_logging():
        from .logging_config import configure_from_environment
        configure_from_environment()
    
    async def validate_environment():
        from .security import SecurityValidator
        validator = SecurityValidator()
        issues = validator.validate_environment_variables()
        if issues:
            for var, issue in issues.items():
                logger.warning(f"Environment validation: {var}: {issue}")
    
    lifecycle_manager.add_startup_task(initialize_logging)
    lifecycle_manager.add_startup_task(validate_environment)
    
    # Add shutdown tasks
    async def save_final_metrics():
        try:
            summary = metrics_collector.get_metrics_summary(hours=1)
            logger.info(f"Final metrics summary: {summary}")
        except Exception as e:
            logger.error(f"Error saving final metrics: {e}")
    
    lifecycle_manager.add_shutdown_task(save_final_metrics)
    
    logger.info("Production lifecycle setup completed")


# Auto-setup when imported
setup_production_lifecycle()