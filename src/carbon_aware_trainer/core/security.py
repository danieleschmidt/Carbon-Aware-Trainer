"""Security utilities and input validation for carbon-aware trainer."""

import os
import re
import hashlib
import secrets
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging

# Optional dependency handling
try:
    import yaml
    HAS_YAML = True
except ImportError:
    yaml = None
    HAS_YAML = False

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


class APIKeyManager:
    """Secure API key management with encryption and rotation."""
    
    def __init__(self, encryption_key: Optional[str] = None):
        """Initialize API key manager.
        
        Args:
            encryption_key: Optional encryption key for storing API keys
        """
        self.validator = SecurityValidator()
        self._encryption_key = encryption_key or self._generate_encryption_key()
        self._api_keys: Dict[str, Dict[str, Any]] = {}
        self._key_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Load API keys from environment and secure storage
        self._load_api_keys_from_environment()
    
    def _generate_encryption_key(self) -> str:
        """Generate encryption key for API key storage.
        
        Returns:
            Base64 encoded encryption key
        """
        try:
            from cryptography.fernet import Fernet
            return Fernet.generate_key().decode()
        except ImportError:
            logger.warning("Cryptography not available, using basic encoding")
            return secrets.token_urlsafe(32)
    
    def _encrypt_api_key(self, api_key: str) -> str:
        """Encrypt API key for secure storage.
        
        Args:
            api_key: API key to encrypt
            
        Returns:
            Encrypted API key
        """
        try:
            from cryptography.fernet import Fernet
            f = Fernet(self._encryption_key.encode())
            return f.encrypt(api_key.encode()).decode()
        except ImportError:
            # Fallback to base64 encoding (not secure, but better than plain text)
            import base64
            return base64.b64encode(api_key.encode()).decode()
    
    def _decrypt_api_key(self, encrypted_key: str) -> str:
        """Decrypt API key from secure storage.
        
        Args:
            encrypted_key: Encrypted API key
            
        Returns:
            Decrypted API key
        """
        try:
            from cryptography.fernet import Fernet
            f = Fernet(self._encryption_key.encode())
            return f.decrypt(encrypted_key.encode()).decode()
        except ImportError:
            import base64
            return base64.b64decode(encrypted_key.encode()).decode()
    
    def store_api_key(
        self, 
        provider: str, 
        api_key: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store API key securely.
        
        Args:
            provider: API provider name
            api_key: API key to store
            metadata: Optional metadata (creation date, etc.)
            
        Returns:
            True if stored successfully
        """
        try:
            # Validate API key
            if not self.validator.validate_api_key(api_key):
                raise ValueError(f"Invalid API key format for {provider}")
            
            # Encrypt and store
            encrypted_key = self._encrypt_api_key(api_key)
            
            self._api_keys[provider] = {
                'encrypted_key': encrypted_key,
                'created_at': datetime.now().isoformat(),
                'last_used': None,
                'usage_count': 0
            }
            
            # Store metadata
            self._key_metadata[provider] = metadata or {}
            
            logger.info(f"API key stored for provider: {provider}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store API key for {provider}: {e}")
            return False
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Retrieve API key for provider.
        
        Args:
            provider: API provider name
            
        Returns:
            Decrypted API key or None if not found
        """
        try:
            if provider not in self._api_keys:
                return None
            
            key_data = self._api_keys[provider]
            decrypted_key = self._decrypt_api_key(key_data['encrypted_key'])
            
            # Update usage statistics
            key_data['last_used'] = datetime.now().isoformat()
            key_data['usage_count'] += 1
            
            return decrypted_key
            
        except Exception as e:
            logger.error(f"Failed to retrieve API key for {provider}: {e}")
            return None
    
    def remove_api_key(self, provider: str) -> bool:
        """Remove API key for provider.
        
        Args:
            provider: API provider name
            
        Returns:
            True if removed successfully
        """
        try:
            if provider in self._api_keys:
                del self._api_keys[provider]
            
            if provider in self._key_metadata:
                del self._key_metadata[provider]
            
            logger.info(f"API key removed for provider: {provider}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove API key for {provider}: {e}")
            return False
    
    def list_providers(self) -> List[str]:
        """List available API key providers.
        
        Returns:
            List of provider names
        """
        return list(self._api_keys.keys())
    
    def get_key_metadata(self, provider: str) -> Optional[Dict[str, Any]]:
        """Get metadata for API key.
        
        Args:
            provider: API provider name
            
        Returns:
            Key metadata or None if not found
        """
        key_data = self._api_keys.get(provider, {})
        metadata = self._key_metadata.get(provider, {})
        
        if not key_data:
            return None
        
        return {
            'provider': provider,
            'created_at': key_data.get('created_at'),
            'last_used': key_data.get('last_used'),
            'usage_count': key_data.get('usage_count', 0),
            'metadata': metadata
        }
    
    def check_key_expiration(self, provider: str, max_age_days: int = 90) -> bool:
        """Check if API key needs rotation.
        
        Args:
            provider: API provider name
            max_age_days: Maximum key age in days
            
        Returns:
            True if key needs rotation
        """
        key_data = self._api_keys.get(provider)
        if not key_data or not key_data.get('created_at'):
            return True
        
        try:
            created_at = datetime.fromisoformat(key_data['created_at'])
            age = datetime.now() - created_at
            return age.days > max_age_days
        except Exception:
            return True
    
    def _load_api_keys_from_environment(self) -> None:
        """Load API keys from environment variables."""
        env_key_mappings = {
            'ELECTRICITYMAP_API_KEY': 'electricitymap',
            'WATTTIME_API_KEY': 'watttime',
            'CARBON_AWARE_API_KEY': 'custom'
        }
        
        for env_var, provider in env_key_mappings.items():
            api_key = os.getenv(env_var)
            if api_key:
                self.store_api_key(provider, api_key, {
                    'source': 'environment',
                    'env_var': env_var
                })
                logger.info(f"Loaded API key for {provider} from environment")
    
    def validate_all_keys(self) -> Dict[str, bool]:
        """Validate all stored API keys.
        
        Returns:
            Dictionary mapping provider to validation status
        """
        results = {}
        
        for provider in self.list_providers():
            api_key = self.get_api_key(provider)
            if api_key:
                results[provider] = self.validator.validate_api_key(api_key)
            else:
                results[provider] = False
        
        return results
    
    def get_key_status_summary(self) -> Dict[str, Any]:
        """Get summary of all API key statuses.
        
        Returns:
            Summary of key statuses
        """
        providers = self.list_providers()
        validation_results = self.validate_all_keys()
        
        summary = {
            'total_keys': len(providers),
            'valid_keys': sum(validation_results.values()),
            'invalid_keys': len(providers) - sum(validation_results.values()),
            'providers': {},
            'expiration_warnings': []
        }
        
        for provider in providers:
            metadata = self.get_key_metadata(provider)
            needs_rotation = self.check_key_expiration(provider)
            
            summary['providers'][provider] = {
                'valid': validation_results.get(provider, False),
                'needs_rotation': needs_rotation,
                'last_used': metadata.get('last_used') if metadata else None,
                'usage_count': metadata.get('usage_count', 0) if metadata else 0
            }
            
            if needs_rotation:
                summary['expiration_warnings'].append(provider)
        
        return summary


class SecureConfigManager:
    """Secure configuration management with encryption support."""
    
    def __init__(self):
        self.validator = SecurityValidator()
        self.api_key_manager = APIKeyManager()
    
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
                if not HAS_YAML:
                    logger.error(f"YAML support not available (install PyYAML), cannot load {path.suffix} file")
                    return None
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
    
    def save_config(
        self, 
        config: Dict[str, Any], 
        config_path: Union[str, Path],
        encrypt_sensitive: bool = True
    ) -> bool:
        """Securely save configuration file with optional encryption.
        
        Args:
            config: Configuration to save
            config_path: Path to save configuration
            encrypt_sensitive: Whether to encrypt sensitive fields
            
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
            
            # Prepare config for saving
            config_to_save = config.copy()
            
            # Handle sensitive fields
            if encrypt_sensitive:
                config_to_save = self._encrypt_sensitive_config_fields(config_to_save)
            else:
                # Remove sensitive fields for plain text storage
                config_to_save = self._remove_sensitive_config_fields(config_to_save)
            
            # Write with secure permissions
            if path.suffix.lower() == '.json':
                import json
                content = json.dumps(config_to_save, indent=2)
            elif path.suffix.lower() in ['.yml', '.yaml']:
                if not HAS_YAML:
                    logger.error(f"YAML support not available (install PyYAML), cannot save {path.suffix} file")
                    return False
                content = yaml.dump(config_to_save, default_flow_style=False)
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
    
    def _encrypt_sensitive_config_fields(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive configuration fields.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with encrypted sensitive fields
        """
        try:
            from cryptography.fernet import Fernet
            
            # Generate or get encryption key
            key = Fernet.generate_key()
            f = Fernet(key)
            
            encrypted_config = config.copy()
            
            # Encrypt API keys
            if 'api_keys' in encrypted_config:
                api_keys = encrypted_config['api_keys']
                encrypted_api_keys = {}
                
                for provider, api_key in api_keys.items():
                    if isinstance(api_key, str) and api_key:
                        encrypted_api_keys[provider] = f.encrypt(api_key.encode()).decode()
                    else:
                        encrypted_api_keys[provider] = api_key
                
                encrypted_config['api_keys'] = encrypted_api_keys
                encrypted_config['_encryption_key'] = key.decode()
                encrypted_config['_encrypted_fields'] = ['api_keys']
            
            return encrypted_config
            
        except ImportError:
            logger.warning("Cryptography not available, cannot encrypt sensitive fields")
            return self._remove_sensitive_config_fields(config)
    
    def _remove_sensitive_config_fields(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive fields from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with sensitive fields removed
        """
        safe_config = config.copy()
        
        # Remove API keys
        if 'api_keys' in safe_config:
            safe_config['api_keys'] = {
                provider: '[REMOVED_FOR_SECURITY]' 
                for provider in safe_config['api_keys']
            }
        
        # Remove other sensitive fields
        sensitive_fields = ['password', 'secret', 'token', 'credential']
        for field in sensitive_fields:
            if field in safe_config:
                safe_config[field] = '[REMOVED_FOR_SECURITY]'
        
        return safe_config