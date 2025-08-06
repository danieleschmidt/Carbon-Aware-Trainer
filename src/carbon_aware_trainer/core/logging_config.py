"""Secure logging configuration for carbon-aware trainer."""

import os
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime

from .security import SecurityValidator


class SecureFormatter(logging.Formatter):
    """Formatter that sanitizes sensitive data in log messages."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validator = SecurityValidator()
    
    def format(self, record):
        """Format log record with security sanitization."""
        # Sanitize the message
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            record.msg = self._sanitize_message(record.msg)
        
        # Sanitize any dictionary arguments
        if hasattr(record, 'args') and record.args:
            sanitized_args = []
            for arg in record.args:
                if isinstance(arg, dict):
                    sanitized_args.append(self.validator.sanitize_log_data(arg))
                else:
                    sanitized_args.append(arg)
            record.args = tuple(sanitized_args)
        
        return super().format(record)
    
    def _sanitize_message(self, message: str) -> str:
        """Sanitize log message to remove sensitive data."""
        import re
        
        # Pattern to match API keys in messages
        api_key_pattern = r'(api[_-]?key["\']?\s*[:=]\s*["\']?)([a-zA-Z0-9_\-\.]{8,})'
        message = re.sub(api_key_pattern, r'\1[REDACTED]', message, flags=re.IGNORECASE)
        
        # Pattern to match tokens
        token_pattern = r'(token["\']?\s*[:=]\s*["\']?)([a-zA-Z0-9_\-\.]{8,})'
        message = re.sub(token_pattern, r'\1[REDACTED]', message, flags=re.IGNORECASE)
        
        # Pattern to match passwords
        password_pattern = r'(password["\']?\s*[:=]\s*["\']?)([^\s"\']+)'
        message = re.sub(password_pattern, r'\1[REDACTED]', message, flags=re.IGNORECASE)
        
        return message


class StructuredFormatter(SecureFormatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        """Format record as JSON with structured fields."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add any extra fields
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info', 'exc_text'):
                if isinstance(value, dict):
                    log_entry[key] = self.validator.sanitize_log_data(value)
                else:
                    log_entry[key] = value
        
        return json.dumps(log_entry)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    structured: bool = False,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """Set up secure logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, logs to console only)
        structured: Whether to use structured JSON logging
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep
    """
    # Validate log level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Choose formatter
    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = SecureFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use rotating file handler to prevent log files from growing too large
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # Set secure file permissions
        log_path.chmod(0o600)
    
    # Configure specific loggers
    configure_library_loggers()
    
    # Log configuration
    root_logger.info("Logging configured", extra={
        'log_level': log_level,
        'log_file': log_file,
        'structured': structured
    })


def configure_library_loggers():
    """Configure logging for third-party libraries."""
    # Reduce verbosity of HTTP libraries
    logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)
    logging.getLogger('aiohttp.access').setLevel(logging.WARNING)
    
    # Reduce verbosity of asyncio
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    
    # Set carbon-aware trainer components to appropriate levels
    logging.getLogger('carbon_aware_trainer.core').setLevel(logging.INFO)
    logging.getLogger('carbon_aware_trainer.carbon_models').setLevel(logging.INFO)
    logging.getLogger('carbon_aware_trainer.strategies').setLevel(logging.INFO)


class AuditLogger:
    """Specialized logger for security-relevant events."""
    
    def __init__(self, log_file: Optional[str] = None):
        """Initialize audit logger.
        
        Args:
            log_file: Path to audit log file
        """
        self.logger = logging.getLogger('carbon_aware_trainer.audit')
        self.logger.setLevel(logging.INFO)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
        
        formatter = StructuredFormatter()
        
        if log_file:
            handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=50 * 1024 * 1024,  # 50MB
                backupCount=10
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
            # Secure permissions
            Path(log_file).chmod(0o600)
    
    def log_authentication_attempt(self, provider: str, success: bool, details: Dict[str, Any] = None):
        """Log authentication attempt.
        
        Args:
            provider: Authentication provider
            success: Whether authentication succeeded
            details: Additional details (will be sanitized)
        """
        sanitized_details = SecurityValidator.sanitize_log_data(details or {})
        
        self.logger.info(
            f"Authentication attempt: {provider}",
            extra={
                'event_type': 'authentication',
                'provider': provider,
                'success': success,
                'details': sanitized_details
            }
        )
    
    def log_configuration_change(self, config_type: str, changes: Dict[str, Any]):
        """Log configuration changes.
        
        Args:
            config_type: Type of configuration changed
            changes: Dictionary of changes made
        """
        sanitized_changes = SecurityValidator.sanitize_log_data(changes)
        
        self.logger.info(
            f"Configuration changed: {config_type}",
            extra={
                'event_type': 'config_change',
                'config_type': config_type,
                'changes': sanitized_changes
            }
        )
    
    def log_security_event(self, event_type: str, severity: str, details: Dict[str, Any]):
        """Log security-relevant events.
        
        Args:
            event_type: Type of security event
            severity: Severity level
            details: Event details
        """
        sanitized_details = SecurityValidator.sanitize_log_data(details)
        
        level = getattr(logging, severity.upper(), logging.INFO)
        
        self.logger.log(
            level,
            f"Security event: {event_type}",
            extra={
                'event_type': 'security_event',
                'security_event_type': event_type,
                'severity': severity,
                'details': sanitized_details
            }
        )


def get_logger(name: str) -> logging.Logger:
    """Get logger with secure configuration.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# Environment-based configuration
def configure_from_environment():
    """Configure logging based on environment variables."""
    log_level = os.getenv('CARBON_AWARE_LOG_LEVEL', 'INFO')
    log_file = os.getenv('CARBON_AWARE_LOG_FILE')
    structured = os.getenv('CARBON_AWARE_STRUCTURED_LOGS', '').lower() in ('true', '1', 'yes')
    
    setup_logging(
        log_level=log_level,
        log_file=log_file,
        structured=structured
    )
    
    # Set up audit logging if specified
    audit_log_file = os.getenv('CARBON_AWARE_AUDIT_LOG_FILE')
    if audit_log_file:
        audit_logger = AuditLogger(audit_log_file)
        # Store globally for access
        globals()['_audit_logger'] = audit_logger


def get_audit_logger() -> Optional[AuditLogger]:
    """Get audit logger instance if configured.
    
    Returns:
        Audit logger or None if not configured
    """
    return globals().get('_audit_logger')


class PerformanceLogger:
    """Logger for performance and timing metrics."""
    
    def __init__(self):
        """Initialize performance logger."""
        self.logger = logging.getLogger('carbon_aware_trainer.performance')
        self.logger.setLevel(logging.INFO)
    
    def log_api_call(self, provider: str, duration_ms: float, success: bool, 
                    region: str = None, error: str = None) -> None:
        """Log API call performance.
        
        Args:
            provider: API provider name
            duration_ms: Call duration in milliseconds  
            success: Whether call was successful
            region: Optional region code
            error: Optional error message
        """
        extra_data = {
            'event_type': 'api_call',
            'provider': provider,
            'duration_ms': duration_ms,
            'success': success
        }
        
        if region:
            extra_data['region'] = region
        if error:
            extra_data['error'] = error
        
        if success:
            self.logger.info(f"API call to {provider} completed in {duration_ms:.1f}ms", 
                           extra=extra_data)
        else:
            self.logger.warning(f"API call to {provider} failed after {duration_ms:.1f}ms: {error}",
                              extra=extra_data)
    
    def log_training_performance(self, samples_per_second: float, carbon_per_sample: float,
                               epoch: int = None, batch: int = None) -> None:
        """Log training performance metrics.
        
        Args:
            samples_per_second: Training throughput
            carbon_per_sample: Carbon cost per training sample
            epoch: Optional epoch number
            batch: Optional batch number
        """
        extra_data = {
            'event_type': 'training_performance',
            'samples_per_second': samples_per_second,
            'carbon_per_sample': carbon_per_sample
        }
        
        if epoch is not None:
            extra_data['epoch'] = epoch
        if batch is not None:
            extra_data['batch'] = batch
        
        self.logger.info(f"Training performance: {samples_per_second:.1f} samples/s, "
                        f"{carbon_per_sample:.4f} gCO2/sample", extra=extra_data)
    
    def log_carbon_decision(self, action: str, current_intensity: float, 
                          threshold: float, region: str) -> None:
        """Log carbon-aware training decisions.
        
        Args:
            action: Action taken (pause, resume, migrate, etc.)
            current_intensity: Current carbon intensity
            threshold: Threshold that triggered the decision
            region: Region code
        """
        extra_data = {
            'event_type': 'carbon_decision',
            'action': action,
            'current_intensity': current_intensity,
            'threshold': threshold,
            'region': region
        }
        
        self.logger.info(f"Carbon decision: {action} (intensity: {current_intensity:.1f}, "
                        f"threshold: {threshold:.1f}, region: {region})", extra=extra_data)


class ErrorTrackingLogger:
    """Logger for error tracking and analysis."""
    
    def __init__(self):
        """Initialize error tracking logger."""
        self.logger = logging.getLogger('carbon_aware_trainer.errors')
        self.logger.setLevel(logging.WARNING)
        self.error_counts = {}
        self.last_reset = datetime.now()
    
    def log_api_error(self, provider: str, error_type: str, error_message: str,
                     region: str = None, retry_attempt: int = None) -> None:
        """Log API errors with context.
        
        Args:
            provider: API provider name
            error_type: Type of error
            error_message: Error message
            region: Optional region code
            retry_attempt: Optional retry attempt number
        """
        error_key = f"{provider}_{error_type}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        extra_data = {
            'event_type': 'api_error',
            'provider': provider,
            'error_type': error_type,
            'error_count': self.error_counts[error_key]
        }
        
        if region:
            extra_data['region'] = region
        if retry_attempt is not None:
            extra_data['retry_attempt'] = retry_attempt
        
        self.logger.error(f"API error in {provider}: {error_type} - {error_message}",
                         extra=extra_data)
    
    def log_training_error(self, error_type: str, error_message: str,
                          epoch: int = None, model_state: str = None) -> None:
        """Log training errors.
        
        Args:
            error_type: Type of training error
            error_message: Error message
            epoch: Optional epoch number where error occurred
            model_state: Optional model state information
        """
        extra_data = {
            'event_type': 'training_error',
            'error_type': error_type
        }
        
        if epoch is not None:
            extra_data['epoch'] = epoch
        if model_state:
            extra_data['model_state'] = model_state
        
        self.logger.error(f"Training error: {error_type} - {error_message}",
                         extra=extra_data)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors.
        
        Returns:
            Dictionary with error statistics
        """
        # Reset counters if it's been more than an hour
        if datetime.now() - self.last_reset > timedelta(hours=1):
            self.error_counts.clear()
            self.last_reset = datetime.now()
        
        return {
            'error_counts': self.error_counts.copy(),
            'total_errors': sum(self.error_counts.values()),
            'reset_time': self.last_reset.isoformat()
        }


class CarbonAuditLogger:
    """Specialized logger for carbon accounting and compliance."""
    
    def __init__(self, log_file: Optional[str] = None):
        """Initialize carbon audit logger.
        
        Args:
            log_file: Path to carbon audit log file
        """
        self.logger = logging.getLogger('carbon_aware_trainer.carbon_audit')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        
        formatter = StructuredFormatter()
        
        if log_file:
            handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=100 * 1024 * 1024,  # 100MB
                backupCount=20
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
            # Secure permissions
            Path(log_file).chmod(0o600)
    
    def log_energy_consumption(self, energy_kwh: float, carbon_kg: float,
                             region: str, training_duration_hours: float,
                             model_info: Dict[str, Any]) -> None:
        """Log energy consumption for carbon accounting.
        
        Args:
            energy_kwh: Energy consumed in kWh
            carbon_kg: Carbon emitted in kg CO2
            region: Region where training occurred
            training_duration_hours: Duration of training
            model_info: Information about the model being trained
        """
        self.logger.info(
            f"Energy consumption recorded: {energy_kwh:.3f} kWh, {carbon_kg:.3f} kg CO2",
            extra={
                'event_type': 'energy_consumption',
                'energy_kwh': energy_kwh,
                'carbon_kg': carbon_kg,
                'region': region,
                'training_duration_hours': training_duration_hours,
                'carbon_intensity_avg': carbon_kg / energy_kwh * 1000 if energy_kwh > 0 else 0,
                'model_info': model_info
            }
        )
    
    def log_carbon_savings(self, baseline_carbon_kg: float, actual_carbon_kg: float,
                          savings_kg: float, optimization_strategy: str,
                          region_original: str, region_optimized: str = None) -> None:
        """Log carbon savings achieved through optimization.
        
        Args:
            baseline_carbon_kg: Baseline carbon emissions
            actual_carbon_kg: Actual carbon emissions
            savings_kg: Carbon savings achieved
            optimization_strategy: Strategy used for optimization
            region_original: Original region
            region_optimized: Optimized region (if migrated)
        """
        savings_percentage = (savings_kg / baseline_carbon_kg * 100) if baseline_carbon_kg > 0 else 0
        
        self.logger.info(
            f"Carbon savings: {savings_kg:.3f} kg CO2 ({savings_percentage:.1f}% reduction)",
            extra={
                'event_type': 'carbon_savings',
                'baseline_carbon_kg': baseline_carbon_kg,
                'actual_carbon_kg': actual_carbon_kg,
                'savings_kg': savings_kg,
                'savings_percentage': savings_percentage,
                'optimization_strategy': optimization_strategy,
                'region_original': region_original,
                'region_optimized': region_optimized
            }
        )
    
    def log_compliance_check(self, carbon_budget_kg: float, consumed_carbon_kg: float,
                           compliance_status: str, reporting_period: str) -> None:
        """Log carbon budget compliance status.
        
        Args:
            carbon_budget_kg: Allocated carbon budget
            consumed_carbon_kg: Carbon consumed so far
            compliance_status: Compliance status (compliant, warning, exceeded)
            reporting_period: Reporting period (monthly, quarterly, etc.)
        """
        budget_utilization = (consumed_carbon_kg / carbon_budget_kg * 100) if carbon_budget_kg > 0 else 0
        
        self.logger.info(
            f"Carbon budget compliance: {compliance_status} ({budget_utilization:.1f}% utilized)",
            extra={
                'event_type': 'compliance_check',
                'carbon_budget_kg': carbon_budget_kg,
                'consumed_carbon_kg': consumed_carbon_kg,
                'remaining_budget_kg': carbon_budget_kg - consumed_carbon_kg,
                'budget_utilization_percent': budget_utilization,
                'compliance_status': compliance_status,
                'reporting_period': reporting_period
            }
        )


# Global logger instances
_performance_logger: Optional[PerformanceLogger] = None
_error_logger: Optional[ErrorTrackingLogger] = None
_carbon_audit_logger: Optional[CarbonAuditLogger] = None


def get_performance_logger() -> PerformanceLogger:
    """Get performance logger instance."""
    global _performance_logger
    if _performance_logger is None:
        _performance_logger = PerformanceLogger()
    return _performance_logger


def get_error_logger() -> ErrorTrackingLogger:
    """Get error tracking logger instance."""
    global _error_logger
    if _error_logger is None:
        _error_logger = ErrorTrackingLogger()
    return _error_logger


def get_carbon_audit_logger() -> CarbonAuditLogger:
    """Get carbon audit logger instance."""
    global _carbon_audit_logger
    if _carbon_audit_logger is None:
        audit_log_file = os.getenv('CARBON_AWARE_CARBON_AUDIT_LOG')
        _carbon_audit_logger = CarbonAuditLogger(audit_log_file)
    return _carbon_audit_logger