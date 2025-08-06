"""Input validation and sanitization using Pydantic models."""

import re
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import logging

try:
    from pydantic import BaseModel, Field, validator, root_validator
    from pydantic import ValidationError as PydanticValidationError
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    PydanticValidationError = ValueError
    # Fallback implementations
    class BaseModel:
        def __init__(self, **data):
            for field, value in data.items():
                setattr(self, field, value)
    
    def Field(default=None, **kwargs):
        return default
    
    def validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def root_validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from .exceptions import ConfigurationError, CarbonDataError
from .security import SecurityValidator


logger = logging.getLogger(__name__)


class RegionCodeModel(BaseModel):
    """Validated region code."""
    code: str = Field(..., min_length=2, max_length=10, description="ISO region code")
    
    @validator('code')
    def validate_region_code(cls, v):
        """Validate region code format."""
        if not isinstance(v, str):
            raise ValueError("Region code must be a string")
        
        # Convert to uppercase
        v = v.upper()
        
        # Basic format validation (ISO 3166-1 alpha-2 with optional subdivision)
        if not re.match(r'^[A-Z]{2}(-[A-Z0-9]{1,4})?$', v):
            raise ValueError(f"Invalid region code format: {v}")
        
        return v


class APIKeyModel(BaseModel):
    """Validated API key."""
    key: str = Field(..., min_length=8, max_length=128, description="API key")
    provider: str = Field(..., description="API key provider")
    
    @validator('key')
    def validate_api_key(cls, v):
        """Validate API key format and security."""
        if not isinstance(v, str):
            raise ValueError("API key must be a string")
        
        # Basic security checks
        if len(v) < 16:
            logger.warning("API key is shorter than recommended minimum (16 chars)")
        
        # Check for obviously invalid keys
        if v.lower() in ['test', 'demo', 'example', 'placeholder']:
            raise ValueError("API key appears to be a placeholder")
        
        # Validate character set (alphanumeric plus common symbols)
        if not re.match(r'^[a-zA-Z0-9_\-\.]+$', v):
            raise ValueError("API key contains invalid characters")
        
        return v
    
    @validator('provider')
    def validate_provider(cls, v):
        """Validate provider name."""
        valid_providers = ['electricitymap', 'watttime', 'custom', 'cached']
        if v.lower() not in valid_providers:
            raise ValueError(f"Invalid provider: {v}. Must be one of {valid_providers}")
        
        return v.lower()


class CarbonIntensityModel(BaseModel):
    """Validated carbon intensity value."""
    value: float = Field(..., ge=0.0, le=2000.0, description="Carbon intensity in gCO2/kWh")
    timestamp: Optional[datetime] = Field(default=None, description="Measurement timestamp")
    region: Optional[str] = Field(default=None, description="Region code")
    
    @validator('value')
    def validate_carbon_intensity(cls, v):
        """Validate carbon intensity value."""
        if not isinstance(v, (int, float)):
            raise ValueError("Carbon intensity must be numeric")
        
        # Check for reasonable bounds
        if v < 0:
            raise ValueError("Carbon intensity cannot be negative")
        
        if v > 2000:
            logger.warning(f"Unusually high carbon intensity: {v} gCO2/kWh")
        
        return float(v)


class ThresholdConfigModel(BaseModel):
    """Validated threshold configuration."""
    carbon_threshold: float = Field(100.0, ge=0.0, le=1000.0, description="Max carbon intensity")
    pause_threshold: float = Field(150.0, ge=0.0, le=1000.0, description="Pause threshold")
    resume_threshold: float = Field(80.0, ge=0.0, le=1000.0, description="Resume threshold")
    
    @root_validator
    def validate_threshold_logic(cls, values):
        """Validate threshold relationships."""
        if not HAS_PYDANTIC:
            return values  # Skip validation when pydantic is not available
            
        carbon_threshold = values.get('carbon_threshold', 0)
        pause_threshold = values.get('pause_threshold', 0)
        resume_threshold = values.get('resume_threshold', 0)
        
        if pause_threshold <= resume_threshold:
            raise ValueError("pause_threshold must be greater than resume_threshold")
        
        if carbon_threshold > pause_threshold:
            logger.warning("carbon_threshold is higher than pause_threshold")
        
        return values


class TimeIntervalModel(BaseModel):
    """Validated time interval."""
    seconds: int = Field(..., ge=1, le=86400, description="Interval in seconds")
    
    @validator('seconds')
    def validate_interval(cls, v):
        """Validate time interval."""
        if v < 1:
            raise ValueError("Interval must be at least 1 second")
        
        if v > 86400:  # 24 hours
            raise ValueError("Interval cannot exceed 24 hours")
        
        # Warn for very short intervals
        if v < 30:
            logger.warning(f"Very short interval ({v}s) may cause excessive API calls")
        
        return v


class FilePathModel(BaseModel):
    """Validated file path."""
    path: str = Field(..., description="File system path")
    must_exist: bool = Field(True, description="Whether file must exist")
    must_be_readable: bool = Field(True, description="Whether file must be readable")
    max_size_mb: int = Field(100, description="Maximum file size in MB")
    
    @validator('path')
    def validate_path(cls, v, values):
        """Validate file path security and accessibility."""
        if not isinstance(v, str):
            raise ValueError("Path must be a string")
        
        try:
            path = Path(v).resolve()
        except (OSError, ValueError) as e:
            raise ValueError(f"Invalid path: {e}")
        
        # Security checks
        if '..' in str(path):
            raise ValueError("Path contains directory traversal")
        
        # Check if path is within reasonable bounds
        if path.is_absolute() and len(path.parts) < 2:
            raise ValueError("Path appears to access root filesystem")
        
        # Check existence if required
        must_exist = values.get('must_exist', True)
        if must_exist and not path.exists():
            raise ValueError(f"File does not exist: {path}")
        
        # Check readability
        must_be_readable = values.get('must_be_readable', True)
        if must_exist and must_be_readable and not os.access(path, os.R_OK):
            raise ValueError(f"File is not readable: {path}")
        
        # Check file size
        if path.exists() and path.is_file():
            max_size_mb = values.get('max_size_mb', 100)
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > max_size_mb:
                raise ValueError(f"File too large: {size_mb:.1f}MB > {max_size_mb}MB")
        
        return str(path)


class TrainingConfigModel(BaseModel):
    """Comprehensive training configuration validation."""
    
    # Thresholds
    thresholds: ThresholdConfigModel = Field(default_factory=ThresholdConfigModel)
    
    # Monitoring intervals
    check_interval: TimeIntervalModel = Field(default_factory=lambda: TimeIntervalModel(seconds=300))
    update_interval: TimeIntervalModel = Field(default_factory=lambda: TimeIntervalModel(seconds=300))
    
    # Regions
    preferred_regions: List[RegionCodeModel] = Field(default_factory=list)
    excluded_regions: List[RegionCodeModel] = Field(default_factory=list)
    
    # Data source configuration
    data_source: str = Field('electricitymap', description="Carbon data source")
    api_keys: Dict[str, APIKeyModel] = Field(default_factory=dict)
    
    # Advanced settings
    max_pause_duration_hours: float = Field(6.0, ge=0.1, le=168.0, description="Max pause duration")
    migration_enabled: bool = Field(False, description="Enable cross-region migration")
    forecast_horizon_hours: float = Field(24.0, ge=1.0, le=168.0, description="Forecast window")
    
    # Resilience settings
    retry_attempts: int = Field(3, ge=1, le=10, description="Max retry attempts")
    timeout_seconds: float = Field(30.0, ge=1.0, le=300.0, description="Request timeout")
    circuit_breaker_enabled: bool = Field(True, description="Enable circuit breaker")
    
    @validator('data_source')
    def validate_data_source(cls, v):
        """Validate data source."""
        valid_sources = ['electricitymap', 'watttime', 'custom', 'cached']
        if v.lower() not in valid_sources:
            raise ValueError(f"Invalid data source: {v}")
        return v.lower()
    
    @root_validator
    def validate_region_compatibility(cls, values):
        """Validate region configuration."""
        if not HAS_PYDANTIC:
            return values  # Skip validation when pydantic is not available
            
        preferred = values.get('preferred_regions', [])
        excluded = values.get('excluded_regions', [])
        
        # Convert to sets of region codes for comparison
        preferred_codes = {r.code if hasattr(r, 'code') else r for r in preferred}
        excluded_codes = {r.code if hasattr(r, 'code') else r for r in excluded}
        
        # Check for conflicts
        overlap = preferred_codes & excluded_codes
        if overlap:
            raise ValueError(f"Regions cannot be both preferred and excluded: {overlap}")
        
        return values


class MonitoringConfigModel(BaseModel):
    """Monitoring and alerting configuration."""
    
    # Health monitoring
    health_check_interval: TimeIntervalModel = Field(
        default_factory=lambda: TimeIntervalModel(seconds=300)
    )
    
    # Alert thresholds
    cpu_threshold_percent: float = Field(90.0, ge=0.0, le=100.0)
    memory_threshold_percent: float = Field(85.0, ge=0.0, le=100.0)
    disk_threshold_percent: float = Field(95.0, ge=0.0, le=100.0)
    response_time_threshold_ms: float = Field(5000.0, ge=100.0, le=60000.0)
    error_rate_threshold_percent: float = Field(10.0, ge=0.0, le=100.0)
    
    # Logging configuration
    log_level: str = Field('INFO', description="Logging level")
    log_file: Optional[FilePathModel] = Field(None, description="Log file path")
    structured_logging: bool = Field(False, description="Enable JSON logging")
    audit_logging: bool = Field(True, description="Enable audit logging")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}")
        return v.upper()


class SecurityConfigModel(BaseModel):
    """Security configuration validation."""
    
    # Authentication
    require_api_keys: bool = Field(True, description="Require API keys for providers")
    api_key_rotation_days: int = Field(90, ge=1, le=365, description="API key rotation period")
    
    # Network security
    ssl_verify: bool = Field(True, description="Verify SSL certificates")
    allowed_hosts: List[str] = Field(default_factory=list, description="Allowed API hosts")
    
    # File security
    config_file_permissions: str = Field('0600', description="Config file permissions")
    log_file_permissions: str = Field('0600', description="Log file permissions")
    
    # Rate limiting
    rate_limit_per_minute: float = Field(60.0, ge=1.0, le=1000.0, description="API rate limit")
    
    @validator('config_file_permissions', 'log_file_permissions')
    def validate_file_permissions(cls, v):
        """Validate file permissions format."""
        if not re.match(r'^[0-7]{3,4}$', v):
            raise ValueError(f"Invalid permission format: {v}")
        
        # Convert to octal for security check
        try:
            perm_int = int(v, 8)
            # Warn if world-readable
            if perm_int & 0o044:  # World or group readable
                logger.warning(f"Permissive file permissions: {v}")
        except ValueError:
            raise ValueError(f"Invalid permission value: {v}")
        
        return v
    
    @validator('allowed_hosts')
    def validate_allowed_hosts(cls, v):
        """Validate allowed hosts list."""
        if not v:
            return v
        
        validated_hosts = []
        for host in v:
            if not isinstance(host, str):
                raise ValueError("Host must be string")
            
            # Basic hostname/URL validation
            if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9\-\.]*[a-zA-Z0-9]$', host):
                raise ValueError(f"Invalid hostname: {host}")
            
            validated_hosts.append(host.lower())
        
        return validated_hosts


class ValidationManager:
    """Centralized validation management."""
    
    def __init__(self):
        """Initialize validation manager."""
        self.security_validator = SecurityValidator()
        self._validation_errors: List[str] = []
    
    def validate_training_config(self, config: Dict[str, Any]) -> TrainingConfigModel:
        """Validate training configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Validated configuration model
            
        Raises:
            ConfigurationError: If validation fails
        """
        try:
            if not HAS_PYDANTIC:
                logger.warning("Pydantic not available, using basic validation")
                return self._basic_training_config_validation(config)
            
            return TrainingConfigModel(**config)
            
        except PydanticValidationError as e:
            error_msg = f"Training configuration validation failed: {e}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg)
    
    def validate_monitoring_config(self, config: Dict[str, Any]) -> MonitoringConfigModel:
        """Validate monitoring configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Validated configuration model
            
        Raises:
            ConfigurationError: If validation fails
        """
        try:
            if not HAS_PYDANTIC:
                return self._basic_monitoring_config_validation(config)
            
            return MonitoringConfigModel(**config)
            
        except PydanticValidationError as e:
            error_msg = f"Monitoring configuration validation failed: {e}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg)
    
    def validate_security_config(self, config: Dict[str, Any]) -> SecurityConfigModel:
        """Validate security configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Validated configuration model
            
        Raises:
            ConfigurationError: If validation fails
        """
        try:
            if not HAS_PYDANTIC:
                return self._basic_security_config_validation(config)
            
            return SecurityConfigModel(**config)
            
        except PydanticValidationError as e:
            error_msg = f"Security configuration validation failed: {e}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg)
    
    def validate_carbon_data(self, data: Dict[str, Any]) -> bool:
        """Validate carbon intensity data.
        
        Args:
            data: Carbon data dictionary
            
        Returns:
            True if valid
            
        Raises:
            CarbonDataError: If validation fails
        """
        try:
            required_fields = ['carbon_intensity', 'timestamp', 'region']
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                raise CarbonDataError(f"Missing required fields: {missing_fields}")
            
            # Validate carbon intensity
            if not HAS_PYDANTIC:
                if not isinstance(data['carbon_intensity'], (int, float)):
                    raise CarbonDataError("Carbon intensity must be numeric")
                if data['carbon_intensity'] < 0:
                    raise CarbonDataError("Carbon intensity cannot be negative")
            else:
                CarbonIntensityModel(value=data['carbon_intensity'])
            
            # Validate timestamp
            if isinstance(data['timestamp'], str):
                try:
                    datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                except ValueError as e:
                    raise CarbonDataError(f"Invalid timestamp format: {e}")
            
            # Validate region
            if not self.security_validator.validate_region_code(data['region']):
                raise CarbonDataError(f"Invalid region code: {data['region']}")
            
            return True
            
        except Exception as e:
            if isinstance(e, CarbonDataError):
                raise
            raise CarbonDataError(f"Carbon data validation failed: {e}")
    
    def _basic_training_config_validation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Basic validation when Pydantic is not available."""
        errors = []
        
        # Validate thresholds
        if 'thresholds' in config:
            threshold_errors = self.security_validator.validate_threshold_values(config['thresholds'])
            errors.extend(threshold_errors)
        
        # Validate regions
        for region_list in ['preferred_regions', 'excluded_regions']:
            if region_list in config:
                for region in config[region_list]:
                    if not self.security_validator.validate_region_code(region):
                        errors.append(f"Invalid region code in {region_list}: {region}")
        
        if errors:
            raise ConfigurationError(f"Validation errors: {'; '.join(errors)}")
        
        return config
    
    def _basic_monitoring_config_validation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Basic monitoring config validation when Pydantic is not available."""
        # Basic bounds checking
        if 'cpu_threshold_percent' in config:
            if not 0 <= config['cpu_threshold_percent'] <= 100:
                raise ConfigurationError("cpu_threshold_percent must be 0-100")
        
        if 'log_level' in config:
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if config['log_level'].upper() not in valid_levels:
                raise ConfigurationError(f"Invalid log level: {config['log_level']}")
        
        return config
    
    def _basic_security_config_validation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Basic security config validation when Pydantic is not available."""
        if 'allowed_hosts' in config:
            for host in config['allowed_hosts']:
                if not isinstance(host, str) or not host:
                    raise ConfigurationError(f"Invalid hostname: {host}")
        
        return config
    
    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors."""
        return self._validation_errors.copy()
    
    def clear_validation_errors(self) -> None:
        """Clear validation error list."""
        self._validation_errors.clear()


# Global validation manager instance
validation_manager = ValidationManager()