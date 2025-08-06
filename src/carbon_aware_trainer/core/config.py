"""Production-ready configuration management system."""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
from datetime import timedelta
import tempfile
import shutil

# Optional dependency handling
try:
    import yaml
    HAS_YAML = True
except ImportError:
    yaml = None
    HAS_YAML = False

from .exceptions import ConfigurationError
from .security import SecureConfigManager, SecurityValidator
from .validation import ValidationManager, validation_manager
from .types import CarbonDataSource, TrainingState


logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration for metrics and state storage."""
    enabled: bool = False
    type: str = "sqlite"  # sqlite, postgresql, mysql
    host: str = "localhost"
    port: int = 5432
    database: str = "carbon_aware_trainer"
    username: str = ""
    password: str = ""
    connection_pool_size: int = 5
    ssl_required: bool = False


@dataclass
class CacheConfig:
    """Caching configuration."""
    enabled: bool = True
    type: str = "memory"  # memory, redis, file
    ttl_seconds: int = 3600
    max_size_mb: int = 100
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    file_cache_dir: str = ""


@dataclass
class AlertingConfig:
    """Alerting and notification configuration."""
    enabled: bool = False
    email_enabled: bool = False
    webhook_enabled: bool = False
    slack_enabled: bool = False
    
    # Email settings
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    email_recipients: List[str] = field(default_factory=list)
    
    # Webhook settings
    webhook_url: str = ""
    webhook_secret: str = ""
    
    # Slack settings
    slack_token: str = ""
    slack_channel: str = ""
    
    # Alert thresholds
    critical_carbon_threshold: float = 500.0
    high_error_rate_threshold: float = 10.0
    system_resource_threshold: float = 90.0


@dataclass
class MetricsConfig:
    """Metrics collection and export configuration."""
    enabled: bool = True
    export_interval_seconds: int = 60
    
    # Export destinations
    prometheus_enabled: bool = False
    prometheus_port: int = 8000
    
    influxdb_enabled: bool = False
    influxdb_url: str = ""
    influxdb_token: str = ""
    influxdb_bucket: str = "carbon-aware-trainer"
    
    # CloudWatch
    cloudwatch_enabled: bool = False
    cloudwatch_region: str = "us-east-1"
    cloudwatch_namespace: str = "CarbonAwareTrainer"
    
    # Custom metrics endpoint
    custom_endpoint_enabled: bool = False
    custom_endpoint_url: str = ""
    custom_endpoint_headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class BackupConfig:
    """Backup and recovery configuration."""
    enabled: bool = True
    backup_interval_hours: int = 24
    retention_days: int = 30
    
    # Backup destinations
    local_backup_dir: str = "./backups"
    s3_backup_enabled: bool = False
    s3_bucket: str = ""
    s3_region: str = "us-east-1"
    
    # What to backup
    backup_configs: bool = True
    backup_metrics: bool = True
    backup_logs: bool = False


@dataclass
class ProductionConfig:
    """Comprehensive production configuration."""
    
    # Basic settings
    environment: str = "production"
    debug: bool = False
    
    # Training configuration
    carbon_threshold: float = 100.0
    pause_threshold: float = 150.0
    resume_threshold: float = 80.0
    check_interval: int = 300
    max_pause_duration_hours: float = 6.0
    
    # Data source
    data_source: CarbonDataSource = CarbonDataSource.ELECTRICITYMAP
    api_keys: Dict[str, str] = field(default_factory=dict)
    
    # Regions
    preferred_regions: List[str] = field(default_factory=list)
    excluded_regions: List[str] = field(default_factory=list)
    migration_enabled: bool = False
    
    # Resilience settings
    retry_max_attempts: int = 3
    retry_base_delay: float = 1.0
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: float = 60.0
    
    # Rate limiting
    rate_limit_requests_per_minute: float = 60.0
    rate_limit_burst: int = 10
    rate_limit_adaptive: bool = True
    
    # Timeouts
    api_timeout_seconds: float = 30.0
    health_check_timeout_seconds: float = 10.0
    
    # Logging
    log_level: str = "INFO"
    log_file: str = ""
    structured_logging: bool = True
    audit_logging: bool = True
    log_rotation_max_size_mb: int = 100
    log_retention_days: int = 30
    
    # Security
    ssl_verify: bool = True
    api_key_validation_enabled: bool = True
    config_encryption_enabled: bool = False
    
    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    alerting: AlertingConfig = field(default_factory=AlertingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    backup: BackupConfig = field(default_factory=BackupConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProductionConfig':
        """Create configuration from dictionary."""
        # Handle nested configurations
        if 'database' in data and isinstance(data['database'], dict):
            data['database'] = DatabaseConfig(**data['database'])
        
        if 'cache' in data and isinstance(data['cache'], dict):
            data['cache'] = CacheConfig(**data['cache'])
        
        if 'alerting' in data and isinstance(data['alerting'], dict):
            data['alerting'] = AlertingConfig(**data['alerting'])
        
        if 'metrics' in data and isinstance(data['metrics'], dict):
            data['metrics'] = MetricsConfig(**data['metrics'])
        
        if 'backup' in data and isinstance(data['backup'], dict):
            data['backup'] = BackupConfig(**data['backup'])
        
        # Handle enum conversion
        if 'data_source' in data and isinstance(data['data_source'], str):
            data['data_source'] = CarbonDataSource(data['data_source'])
        
        return cls(**data)


class ConfigManager:
    """Production-grade configuration management with validation, encryption, and backup."""
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """Initialize configuration manager.
        
        Args:
            config_dir: Configuration directory path
        """
        self.config_dir = Path(config_dir) if config_dir else Path.cwd() / "config"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.secure_config_manager = SecureConfigManager()
        self.security_validator = SecurityValidator()
        self.validation_manager = validation_manager
        
        # Current configuration
        self._config: Optional[ProductionConfig] = None
        self._config_file_path: Optional[Path] = None
        
        # Configuration watchers (for hot reloading)
        self._config_watchers: List[callable] = []
        
        logger.info(f"ConfigManager initialized with config_dir: {self.config_dir}")
    
    def load_config(
        self, 
        config_path: Optional[Union[str, Path]] = None,
        environment: str = "production"
    ) -> ProductionConfig:
        """Load configuration from file with validation.
        
        Args:
            config_path: Path to configuration file
            environment: Environment name for config selection
            
        Returns:
            Loaded and validated configuration
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        if config_path:
            config_file = Path(config_path)
        else:
            config_file = self._find_config_file(environment)
        
        if not config_file or not config_file.exists():
            logger.info("No config file found, using default configuration")
            self._config = ProductionConfig(environment=environment)
            return self._config
        
        logger.info(f"Loading configuration from: {config_file}")
        
        try:
            # Load raw configuration
            raw_config = self.secure_config_manager.load_config(config_file)
            if not raw_config:
                raise ConfigurationError(f"Failed to load config from {config_file}")
            
            # Merge with environment variables
            merged_config = self._merge_environment_variables(raw_config)
            
            # Validate configuration
            validated_config = self._validate_configuration(merged_config)
            
            # Create configuration object
            self._config = ProductionConfig.from_dict(validated_config)
            self._config.environment = environment
            self._config_file_path = config_file
            
            # Perform post-load processing
            self._post_load_processing()
            
            logger.info("Configuration loaded successfully")
            return self._config
            
        except Exception as e:
            error_msg = f"Failed to load configuration: {e}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg)
    
    def save_config(
        self, 
        config: Optional[ProductionConfig] = None,
        config_path: Optional[Union[str, Path]] = None,
        backup_existing: bool = True
    ) -> bool:
        """Save configuration to file.
        
        Args:
            config: Configuration to save (defaults to current)
            config_path: Path to save configuration
            backup_existing: Whether to backup existing config
            
        Returns:
            True if saved successfully
        """
        config_to_save = config or self._config
        if not config_to_save:
            raise ConfigurationError("No configuration to save")
        
        save_path = Path(config_path) if config_path else self._config_file_path
        if not save_path:
            # Default to JSON if YAML is not available
            extension = "json" if not HAS_YAML else "yaml"
            save_path = self.config_dir / f"{config_to_save.environment}.{extension}"
        
        try:
            # Backup existing configuration
            if backup_existing and save_path.exists():
                self._backup_config_file(save_path)
            
            # Convert to dictionary and sanitize
            config_dict = config_to_save.to_dict()
            sanitized_config = self._sanitize_config_for_save(config_dict)
            
            # Save configuration
            success = self.secure_config_manager.save_config(sanitized_config, save_path)
            if success:
                self._config_file_path = save_path
                logger.info(f"Configuration saved to: {save_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def reload_config(self) -> bool:
        """Reload configuration from file.
        
        Returns:
            True if reloaded successfully
        """
        if not self._config_file_path or not self._config_file_path.exists():
            logger.warning("Cannot reload: no config file path")
            return False
        
        try:
            old_config = self._config
            self.load_config(self._config_file_path)
            
            # Notify watchers of config change
            self._notify_config_change(old_config, self._config)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            return False
    
    def get_config(self) -> ProductionConfig:
        """Get current configuration.
        
        Returns:
            Current configuration
            
        Raises:
            ConfigurationError: If no configuration is loaded
        """
        if not self._config:
            raise ConfigurationError("No configuration loaded")
        
        return self._config
    
    def update_config(self, updates: Dict[str, Any], save: bool = True) -> bool:
        """Update configuration with new values.
        
        Args:
            updates: Dictionary of updates
            save: Whether to save after updating
            
        Returns:
            True if updated successfully
        """
        if not self._config:
            raise ConfigurationError("No configuration to update")
        
        try:
            # Apply updates
            config_dict = self._config.to_dict()
            self._apply_config_updates(config_dict, updates)
            
            # Validate updated configuration
            validated_config = self._validate_configuration(config_dict)
            
            # Create new configuration object
            old_config = self._config
            self._config = ProductionConfig.from_dict(validated_config)
            
            # Save if requested
            if save:
                self.save_config()
            
            # Notify watchers
            self._notify_config_change(old_config, self._config)
            
            logger.info("Configuration updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return False
    
    def add_config_watcher(self, callback: callable) -> None:
        """Add configuration change watcher.
        
        Args:
            callback: Function to call when configuration changes
        """
        self._config_watchers.append(callback)
    
    def remove_config_watcher(self, callback: callable) -> None:
        """Remove configuration change watcher.
        
        Args:
            callback: Callback to remove
        """
        if callback in self._config_watchers:
            self._config_watchers.remove(callback)
    
    def validate_current_config(self) -> List[str]:
        """Validate current configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        if not self._config:
            return ["No configuration loaded"]
        
        try:
            config_dict = self._config.to_dict()
            self._validate_configuration(config_dict)
            return []
        except Exception as e:
            return [str(e)]
    
    def get_config_status(self) -> Dict[str, Any]:
        """Get configuration system status.
        
        Returns:
            Dictionary with status information
        """
        return {
            "config_loaded": self._config is not None,
            "config_file": str(self._config_file_path) if self._config_file_path else None,
            "config_dir": str(self.config_dir),
            "environment": self._config.environment if self._config else None,
            "last_loaded": getattr(self._config, '_last_loaded', None),
            "validation_errors": self.validate_current_config(),
            "watchers_count": len(self._config_watchers)
        }
    
    def _find_config_file(self, environment: str) -> Optional[Path]:
        """Find configuration file for environment.
        
        Args:
            environment: Environment name
            
        Returns:
            Path to config file or None if not found
        """
        # Try various file names and extensions
        # Prefer JSON if YAML is not available
        if HAS_YAML:
            candidates = [
                f"{environment}.yaml",
                f"{environment}.yml",
                f"{environment}.json",
                "config.yaml",
                "config.yml",
                "config.json"
            ]
        else:
            candidates = [
                f"{environment}.json",
                "config.json",
                f"{environment}.yaml",  # Will fail gracefully with helpful error
                f"{environment}.yml",
                "config.yaml",
                "config.yml"
            ]
        
        for candidate in candidates:
            config_file = self.config_dir / candidate
            if config_file.exists():
                return config_file
        
        return None
    
    def _merge_environment_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration with environment variables.
        
        Args:
            config: Base configuration
            
        Returns:
            Merged configuration
        """
        # Define environment variable mappings
        env_mappings = {
            'CARBON_AWARE_LOG_LEVEL': 'log_level',
            'CARBON_AWARE_LOG_FILE': 'log_file',
            'CARBON_AWARE_DEBUG': 'debug',
            'CARBON_AWARE_DATA_SOURCE': 'data_source',
            'CARBON_AWARE_CARBON_THRESHOLD': 'carbon_threshold',
            'CARBON_AWARE_CHECK_INTERVAL': 'check_interval',
            'ELECTRICITYMAP_API_KEY': 'api_keys.electricitymap',
            'WATTTIME_API_KEY': 'api_keys.watttime',
            'CARBON_AWARE_SSL_VERIFY': 'ssl_verify',
        }
        
        merged = config.copy()
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Parse environment variable value
                parsed_value = self._parse_env_value(env_value)
                
                # Set nested configuration value
                self._set_nested_value(merged, config_path, parsed_value)
                
                logger.debug(f"Applied environment variable: {env_var} -> {config_path}")
        
        return merged
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type.
        
        Args:
            value: Environment variable value
            
        Returns:
            Parsed value
        """
        # Boolean values
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Numeric values
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # String value
        return value
    
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any) -> None:
        """Set nested configuration value.
        
        Args:
            config: Configuration dictionary
            path: Dot-separated path
            value: Value to set
        """
        keys = path.split('.')
        current = config
        
        # Navigate to parent of target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set final value
        current[keys[-1]] = value
    
    def _validate_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration comprehensively.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Validated configuration
            
        Raises:
            ConfigurationError: If validation fails
        """
        errors = []
        
        # Use ValidationManager for comprehensive validation
        try:
            # Validate security settings
            security_config = config.get('security', {})
            if security_config:
                self.validation_manager.validate_security_config(security_config)
            
            # Validate thresholds
            if 'carbon_threshold' in config:
                thresholds = {
                    'carbon_threshold': config['carbon_threshold'],
                    'pause_threshold': config.get('pause_threshold', 150.0),
                    'resume_threshold': config.get('resume_threshold', 80.0)
                }
                threshold_errors = self.security_validator.validate_threshold_values(thresholds)
                errors.extend(threshold_errors)
            
            # Validate regions
            for region_key in ['preferred_regions', 'excluded_regions']:
                regions = config.get(region_key, [])
                for region in regions:
                    if not self.security_validator.validate_region_code(region):
                        errors.append(f"Invalid region in {region_key}: {region}")
            
            # Validate API keys
            api_keys = config.get('api_keys', {})
            for provider, api_key in api_keys.items():
                if api_key and not self.security_validator.validate_api_key(api_key):
                    errors.append(f"Invalid API key format for {provider}")
            
            # Validate file paths
            for path_key in ['log_file']:
                if path_key in config and config[path_key]:
                    path_value = config[path_key]
                    if not self.security_validator.validate_file_path(path_value, must_exist=False):
                        errors.append(f"Invalid file path for {path_key}: {path_value}")
            
            if errors:
                raise ConfigurationError(f"Configuration validation failed: {'; '.join(errors)}")
            
            return config
            
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Configuration validation error: {e}")
    
    def _post_load_processing(self) -> None:
        """Perform post-load configuration processing."""
        if not self._config:
            return
        
        # Set up cache directory
        if self._config.cache.enabled and self._config.cache.type == "file":
            if not self._config.cache.file_cache_dir:
                self._config.cache.file_cache_dir = str(self.config_dir / "cache")
            
            cache_dir = Path(self._config.cache.file_cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up backup directory
        if self._config.backup.enabled:
            backup_dir = Path(self._config.backup.local_backup_dir)
            backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate environment-specific settings
        if self._config.environment == "production":
            self._validate_production_settings()
    
    def _validate_production_settings(self) -> None:
        """Validate production-specific settings."""
        warnings = []
        
        if self._config.debug:
            warnings.append("Debug mode enabled in production")
        
        if not self._config.ssl_verify:
            warnings.append("SSL verification disabled in production")
        
        if not self._config.audit_logging:
            warnings.append("Audit logging disabled in production")
        
        if self._config.log_level == "DEBUG":
            warnings.append("Debug logging enabled in production")
        
        for warning in warnings:
            logger.warning(f"Production configuration warning: {warning}")
    
    def _sanitize_config_for_save(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize configuration before saving.
        
        Args:
            config: Configuration to sanitize
            
        Returns:
            Sanitized configuration
        """
        sanitized = config.copy()
        
        # Remove runtime-only fields
        runtime_fields = ['_last_loaded', '_config_watchers']
        for field in runtime_fields:
            sanitized.pop(field, None)
        
        # Sanitize sensitive data for logging
        if 'api_keys' in sanitized:
            sanitized['api_keys'] = self.security_validator.sanitize_log_data(sanitized['api_keys'])
        
        return sanitized
    
    def _apply_config_updates(self, config: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Apply updates to configuration dictionary.
        
        Args:
            config: Configuration to update
            updates: Updates to apply
        """
        for key, value in updates.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                # Recursively update nested dictionaries
                self._apply_config_updates(config[key], value)
            else:
                config[key] = value
    
    def _backup_config_file(self, config_file: Path) -> None:
        """Backup configuration file.
        
        Args:
            config_file: Configuration file to backup
        """
        if not config_file.exists():
            return
        
        backup_dir = self.config_dir / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{config_file.stem}_{timestamp}{config_file.suffix}"
        backup_path = backup_dir / backup_name
        
        try:
            shutil.copy2(config_file, backup_path)
            logger.info(f"Configuration backed up to: {backup_path}")
        except Exception as e:
            logger.warning(f"Failed to backup configuration: {e}")
    
    def _notify_config_change(
        self, 
        old_config: Optional[ProductionConfig], 
        new_config: ProductionConfig
    ) -> None:
        """Notify watchers of configuration change.
        
        Args:
            old_config: Previous configuration
            new_config: New configuration
        """
        for watcher in self._config_watchers:
            try:
                watcher(old_config, new_config)
            except Exception as e:
                logger.error(f"Config watcher error: {e}")


# Global configuration manager
config_manager = ConfigManager()