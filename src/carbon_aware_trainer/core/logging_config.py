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