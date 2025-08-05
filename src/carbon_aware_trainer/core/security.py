"""Security utilities and input validation for carbon-aware trainer."""

import os
import re
import hashlib
import secrets
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SecurityValidator:
    """Security validation and sanitization utilities."""
    
    # Allowed characters for different input types
    REGION_CODE_PATTERN = re.compile(r'^[A-Z]{2}(-[A-Z0-9]{1,4})?$')
    API_KEY_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.]{8,128}$')
    SESSION_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_\-]{8,64}$')
    
    # Sensitive field names to redact in logs
    SENSITIVE_FIELDS = {
        'api_key', 'password', 'secret', 'token', 'auth',
        'credential', 'private_key', 'access_token'
    }
    
    @classmethod
    def validate_region_code(cls, region: str) -> bool:
        """Validate region code format.
        
        Args:
            region: Region code to validate
            
        Returns:
            True if valid region code format
        """
        if not isinstance(region, str):
            return False
        
        return bool(cls.REGION_CODE_PATTERN.match(region))
    
    @classmethod
    def validate_api_key(cls, api_key: str) -> bool:
        """Validate API key format.
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if valid API key format
        """
        if not isinstance(api_key, str):
            return False
        
        # Check basic format
        if not cls.API_KEY_PATTERN.match(api_key):
            return False
        
        # Additional security checks
        if len(api_key) < 16:  # Minimum length for security
            return False
        
        return True
    
    @classmethod
    def validate_file_path(cls, file_path: Union[str, Path], must_exist: bool = True) -> bool:
        """Validate file path for security.
        
        Args:
            file_path: File path to validate
            must_exist: Whether file must exist
            
        Returns:
            True if path is safe
        """
        try:
            path = Path(file_path).resolve()
            
            # Check for path traversal attempts
            if '..' in str(path):
                return False
            
            # Check if path is within reasonable bounds (not root filesystem)
            if path.is_absolute() and len(path.parts) < 2:
                return False
            
            # Check existence if required
            if must_exist and not path.exists():
                return False
            
            # Check if it's actually a file (not directory or special file)
            if must_exist and not path.is_file():
                return False
            
            return True
            
        except (OSError, ValueError):
            return False
    
    @classmethod
    def sanitize_log_data(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize data for safe logging by redacting sensitive fields.
        
        Args:
            data: Dictionary to sanitize
            
        Returns:
            Sanitized dictionary with sensitive fields redacted
        """
        if not isinstance(data, dict):
            return data
        
        sanitized = {}
        
        for key, value in data.items():
            key_lower = key.lower()
            
            # Check if field name indicates sensitive data
            is_sensitive = any(
                sensitive in key_lower 
                for sensitive in cls.SENSITIVE_FIELDS
            )
            
            if is_sensitive:
                if isinstance(value, str) and len(value) > 8:
                    # Show first 4 and last 4 characters
                    sanitized[key] = f"{value[:4]}...{value[-4:]}"
                else:
                    sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                # Recursively sanitize nested dictionaries
                sanitized[key] = cls.sanitize_log_data(value)
            else:
                sanitized[key] = value
        
        return sanitized
    
    @classmethod
    def validate_carbon_intensity(cls, intensity: float) -> bool:
        """Validate carbon intensity value.
        
        Args:
            intensity: Carbon intensity value to validate
            
        Returns:
            True if valid carbon intensity
        """
        if not isinstance(intensity, (int, float)):
            return False
        
        # Reasonable bounds for carbon intensity (gCO2/kWh)
        return 0 <= intensity <= 2000
    
    @classmethod
    def validate_threshold_values(cls, thresholds: Dict[str, float]) -> List[str]:
        """Validate threshold configuration values.
        
        Args:
            thresholds: Dictionary of threshold values
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        required_thresholds = ['carbon_threshold', 'pause_threshold', 'resume_threshold']
        
        for threshold_name in required_thresholds:
            if threshold_name not in thresholds:
                errors.append(f"Missing required threshold: {threshold_name}")
                continue
            
            value = thresholds[threshold_name]
            
            if not isinstance(value, (int, float)):
                errors.append(f"{threshold_name} must be numeric")
                continue
            
            if not cls.validate_carbon_intensity(value):
                errors.append(f"{threshold_name} value {value} outside valid range (0-2000)")
        
        # Logical validation
        if 'pause_threshold' in thresholds and 'resume_threshold' in thresholds:
            if thresholds['pause_threshold'] <= thresholds['resume_threshold']:
                errors.append("pause_threshold must be greater than resume_threshold")
        
        return errors
    
    @classmethod
    def generate_session_id(cls) -> str:
        """Generate secure session ID.
        
        Returns:
            Cryptographically secure session ID
        """
        return secrets.token_urlsafe(32)
    
    @classmethod
    def hash_sensitive_data(cls, data: str) -> str:
        """Hash sensitive data for storage or comparison.
        
        Args:
            data: Sensitive data to hash
            
        Returns:
            SHA-256 hash of the data
        """
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    
    @classmethod
    def validate_environment_variables(cls) -> Dict[str, str]:
        """Validate and sanitize environment variables.
        
        Returns:
            Dictionary of validation issues found
        """
        issues = {}
        
        # Check for common insecure environment variables
        dangerous_env_vars = [
            'CARBON_AWARE_DEBUG_LOG_SECRETS',
            'CARBON_AWARE_DISABLE_SSL_VERIFY',
            'CARBON_AWARE_ALLOW_INSECURE'
        ]
        
        for var in dangerous_env_vars:
            if os.getenv(var):
                issues[var] = "Potentially insecure environment variable set"
        
        # Validate API keys if present
        api_key_vars = [
            'ELECTRICITYMAP_API_KEY',
            'WATTTIME_API_KEY',
            'CARBON_AWARE_API_KEY'
        ]
        
        for var in api_key_vars:
            value = os.getenv(var)
            if value and not cls.validate_api_key(value):
                issues[var] = "API key format appears invalid"
        
        return issues
    
    @classmethod
    def sanitize_file_content(cls, content: str, max_size: int = 10_000_000) -> Optional[str]:
        """Sanitize file content for security.
        
        Args:
            content: File content to sanitize
            max_size: Maximum allowed file size in bytes
            
        Returns:
            Sanitized content or None if invalid
        """
        if not isinstance(content, str):
            return None
        
        # Check size limit
        if len(content.encode('utf-8')) > max_size:
            logger.warning(f"File content exceeds size limit ({max_size} bytes)")
            return None
        
        # Remove potentially dangerous content
        # This is a basic implementation - more sophisticated filtering may be needed
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',                # JavaScript URLs
            r'data:.*base64',             # Base64 data URLs
            r'eval\s*\(',                 # eval() calls
        ]
        
        sanitized = content
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, '[REMOVED]', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        return sanitized


class SecureConfigManager:
    """Secure configuration management."""
    
    def __init__(self):
        self.validator = SecurityValidator()
    
    def load_config(self, config_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Securely load configuration file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary or None if invalid
        """
        if not self.validator.validate_file_path(config_path, must_exist=True):
            logger.error(f"Invalid or dangerous config path: {config_path}")
            return None
        
        try:
            path = Path(config_path)
            
            # Check file permissions (should not be world-readable for sensitive configs)
            if path.stat().st_mode & 0o044:  # World or group readable
                logger.warning(f"Config file {config_path} has permissive permissions")
            
            with open(path, 'r') as f:
                content = f.read()
            
            # Sanitize content
            sanitized_content = self.validator.sanitize_file_content(content)
            if not sanitized_content:
                logger.error("Config file content failed security validation")
                return None
            
            # Parse configuration (assuming JSON/YAML)
            if path.suffix.lower() == '.json':
                import json
                config = json.loads(sanitized_content)
            elif path.suffix.lower() in ['.yml', '.yaml']:
                import yaml
                config = yaml.safe_load(sanitized_content)
            else:
                logger.error(f"Unsupported config file format: {path.suffix}")
                return None
            
            # Validate configuration structure
            validation_errors = self._validate_config_structure(config)
            if validation_errors:
                logger.error(f"Config validation errors: {validation_errors}")
                return None
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return None
    
    def _validate_config_structure(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration structure and values.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Validate threshold values if present
        if 'thresholds' in config:
            threshold_errors = self.validator.validate_threshold_values(config['thresholds'])
            errors.extend(threshold_errors)
        
        # Validate regions if present
        if 'regions' in config:
            regions = config['regions']
            if isinstance(regions, list):
                for region in regions:
                    if not self.validator.validate_region_code(region):
                        errors.append(f"Invalid region code: {region}")
        
        # Validate API keys if present (but don't log them)
        if 'api_keys' in config:
            api_keys = config['api_keys']
            if isinstance(api_keys, dict):
                for provider, key in api_keys.items():
                    if not self.validator.validate_api_key(key):
                        errors.append(f"Invalid API key format for provider: {provider}")
        
        return errors
    
    def save_config(self, config: Dict[str, Any], config_path: Union[str, Path]) -> bool:
        """Securely save configuration file.
        
        Args:
            config: Configuration to save
            config_path: Path to save configuration
            
        Returns:
            True if saved successfully
        """
        try:
            path = Path(config_path)
            
            # Validate configuration before saving
            validation_errors = self._validate_config_structure(config)
            if validation_errors:
                logger.error(f"Cannot save invalid config: {validation_errors}")
                return False
            
            # Create directory if needed
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write with secure permissions
            if path.suffix.lower() == '.json':
                import json
                content = json.dumps(config, indent=2)
            elif path.suffix.lower() in ['.yml', '.yaml']:
                import yaml
                content = yaml.dump(config, default_flow_style=False)
            else:
                logger.error(f"Unsupported config file format: {path.suffix}")
                return False
            
            # Write file with restrictive permissions
            with open(path, 'w') as f:
                f.write(content)
            
            # Set secure file permissions (owner read/write only)
            path.chmod(0o600)
            
            logger.info(f"Saved configuration to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {e}")
            return False