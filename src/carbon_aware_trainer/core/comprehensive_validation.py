"""
Comprehensive validation system for carbon-aware training components.
"""

import re
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import logging

from pydantic import BaseModel, validator
import numpy as np

from .types import CarbonIntensity, CarbonForecast, TrainingConfig
from .exceptions import CarbonAwareException

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"


class ValidationRule(Enum):
    """Validation rule types."""
    REQUIRED = "required"
    TYPE_CHECK = "type_check"
    RANGE = "range"
    FORMAT = "format"
    LOGICAL = "logical"
    SECURITY = "security"
    PERFORMANCE = "performance"


@dataclass
class ValidationIssue:
    """A validation issue."""
    field: str
    rule: ValidationRule
    severity: ValidationSeverity
    message: str
    value: Any = None
    suggestion: Optional[str] = None


class ValidationResult:
    """Result of validation process."""
    
    def __init__(self):
        self.issues: List[ValidationIssue] = []
        self.is_valid = True
        
    def add_issue(
        self,
        field: str,
        rule: ValidationRule,
        severity: ValidationSeverity,
        message: str,
        value: Any = None,
        suggestion: Optional[str] = None
    ):
        """Add validation issue."""
        issue = ValidationIssue(field, rule, severity, message, value, suggestion)
        self.issues.append(issue)
        
        if severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
            self.is_valid = False
            
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get issues by severity level."""
        return [issue for issue in self.issues if issue.severity == severity]
        
    def has_errors(self) -> bool:
        """Check if validation has errors or critical issues."""
        return any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                  for issue in self.issues)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "issues": [
                {
                    "field": issue.field,
                    "rule": issue.rule.value,
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "value": str(issue.value) if issue.value is not None else None,
                    "suggestion": issue.suggestion
                }
                for issue in self.issues
            ]
        }


class CarbonDataValidator:
    """Validates carbon intensity data."""
    
    @staticmethod
    def validate_carbon_intensity(data: CarbonIntensity) -> ValidationResult:
        """Validate carbon intensity data."""
        result = ValidationResult()
        
        # Check carbon intensity value
        if data.carbon_intensity is None:
            result.add_issue(
                "carbon_intensity",
                ValidationRule.REQUIRED,
                ValidationSeverity.CRITICAL,
                "Carbon intensity value is required"
            )
        elif not isinstance(data.carbon_intensity, (int, float)):
            result.add_issue(
                "carbon_intensity",
                ValidationRule.TYPE_CHECK,
                ValidationSeverity.ERROR,
                f"Carbon intensity must be numeric, got {type(data.carbon_intensity)}"
            )
        elif data.carbon_intensity < 0:
            result.add_issue(
                "carbon_intensity",
                ValidationRule.RANGE,
                ValidationSeverity.ERROR,
                f"Carbon intensity cannot be negative: {data.carbon_intensity}",
                suggestion="Check data source for errors"
            )
        elif data.carbon_intensity > 1000:
            result.add_issue(
                "carbon_intensity",
                ValidationRule.RANGE,
                ValidationSeverity.WARNING,
                f"Unusually high carbon intensity: {data.carbon_intensity} gCO2/kWh",
                suggestion="Verify data accuracy - typical values are 0-800 gCO2/kWh"
            )
            
        # Check timestamp
        if data.timestamp is None:
            result.add_issue(
                "timestamp",
                ValidationRule.REQUIRED,
                ValidationSeverity.ERROR,
                "Timestamp is required"
            )
        elif not isinstance(data.timestamp, datetime):
            result.add_issue(
                "timestamp",
                ValidationRule.TYPE_CHECK,
                ValidationSeverity.ERROR,
                f"Timestamp must be datetime object, got {type(data.timestamp)}"
            )
        else:
            # Check if timestamp is reasonable
            now = datetime.now()
            age = abs((now - data.timestamp).total_seconds())
            
            if age > 3600:  # Older than 1 hour
                result.add_issue(
                    "timestamp",
                    ValidationRule.LOGICAL,
                    ValidationSeverity.WARNING,
                    f"Carbon data is {age/3600:.1f} hours old",
                    suggestion="Consider refreshing carbon intensity data"
                )
                
        # Check region
        if not data.region:
            result.add_issue(
                "region",
                ValidationRule.REQUIRED,
                ValidationSeverity.ERROR,
                "Region identifier is required"
            )
        elif not isinstance(data.region, str):
            result.add_issue(
                "region",
                ValidationRule.TYPE_CHECK,
                ValidationSeverity.ERROR,
                f"Region must be string, got {type(data.region)}"
            )
            
        return result
        
    @staticmethod
    def validate_forecast(forecast: CarbonForecast) -> ValidationResult:
        """Validate carbon forecast data."""
        result = ValidationResult()
        
        # Check forecast has data points
        if not forecast.data_points:
            result.add_issue(
                "data_points",
                ValidationRule.REQUIRED,
                ValidationSeverity.ERROR,
                "Forecast must contain data points"
            )
            return result
            
        # Validate each data point
        for i, point in enumerate(forecast.data_points):
            point_result = CarbonDataValidator.validate_carbon_intensity(point)
            for issue in point_result.issues:
                result.add_issue(
                    f"data_points[{i}].{issue.field}",
                    issue.rule,
                    issue.severity,
                    issue.message,
                    issue.value,
                    issue.suggestion
                )
                
        # Check forecast temporal consistency
        timestamps = [dp.timestamp for dp in forecast.data_points if dp.timestamp]
        if len(timestamps) > 1:
            # Check if timestamps are in order
            for i in range(1, len(timestamps)):
                if timestamps[i] <= timestamps[i-1]:
                    result.add_issue(
                        "data_points",
                        ValidationRule.LOGICAL,
                        ValidationSeverity.WARNING,
                        f"Forecast timestamps not in chronological order at index {i}",
                        suggestion="Sort forecast data points by timestamp"
                    )
                    break
                    
            # Check for reasonable intervals
            intervals = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                        for i in range(1, len(timestamps))]
            if intervals:
                avg_interval = sum(intervals) / len(intervals)
                if avg_interval < 300:  # Less than 5 minutes
                    result.add_issue(
                        "data_points",
                        ValidationRule.LOGICAL,
                        ValidationSeverity.INFO,
                        f"Very frequent forecast intervals: {avg_interval:.0f}s average",
                        suggestion="Consider aggregating to hourly intervals"
                    )
                elif avg_interval > 14400:  # More than 4 hours
                    result.add_issue(
                        "data_points",
                        ValidationRule.LOGICAL,
                        ValidationSeverity.WARNING,
                        f"Large gaps in forecast data: {avg_interval/3600:.1f}h average interval",
                        suggestion="Interpolate missing data points"
                    )
                    
        return result


class TrainingConfigValidator:
    """Validates training configuration."""
    
    @staticmethod
    def validate_config(config: TrainingConfig) -> ValidationResult:
        """Validate training configuration."""
        result = ValidationResult()
        
        # Carbon threshold validation
        if hasattr(config, 'carbon_threshold') and config.carbon_threshold is not None:
            if config.carbon_threshold < 0:
                result.add_issue(
                    "carbon_threshold",
                    ValidationRule.RANGE,
                    ValidationSeverity.ERROR,
                    f"Carbon threshold cannot be negative: {config.carbon_threshold}"
                )
            elif config.carbon_threshold > 1000:
                result.add_issue(
                    "carbon_threshold",
                    ValidationRule.RANGE,
                    ValidationSeverity.WARNING,
                    f"Very high carbon threshold: {config.carbon_threshold} gCO2/kWh",
                    suggestion="Consider lower threshold for better carbon reduction"
                )
            elif config.carbon_threshold < 50:
                result.add_issue(
                    "carbon_threshold",
                    ValidationRule.PERFORMANCE,
                    ValidationSeverity.WARNING,
                    f"Very low carbon threshold: {config.carbon_threshold} gCO2/kWh",
                    suggestion="This may cause frequent training pauses"
                )
                
        # Batch size validation
        if hasattr(config, 'batch_size') and config.batch_size is not None:
            if config.batch_size < 1:
                result.add_issue(
                    "batch_size",
                    ValidationRule.RANGE,
                    ValidationSeverity.ERROR,
                    f"Batch size must be positive: {config.batch_size}"
                )
            elif config.batch_size > 1024:
                result.add_issue(
                    "batch_size",
                    ValidationRule.PERFORMANCE,
                    ValidationSeverity.WARNING,
                    f"Very large batch size: {config.batch_size}",
                    suggestion="Large batches may cause memory issues"
                )
                
        # Region validation
        if hasattr(config, 'region') and config.region:
            if not CarbonDataValidator.validate_region_code(config.region):
                result.add_issue(
                    "region",
                    ValidationRule.FORMAT,
                    ValidationSeverity.WARNING,
                    f"Non-standard region format: {config.region}",
                    suggestion="Use ISO 3166-2 format (e.g., 'US-CA', 'EU-DE')"
                )
                
        return result
        
    @staticmethod
    def validate_region_code(region: str) -> bool:
        """Validate region code format."""
        # Simple pattern for country-region codes
        pattern = r'^[A-Z]{2}(-[A-Z]{2})?$'
        return bool(re.match(pattern, region))


class SecurityValidator:
    """Security-focused validation."""
    
    DANGEROUS_PATTERNS = [
        r'<script',
        r'javascript:',
        r'onload=',
        r'eval\(',
        r'exec\(',
        r'\${',  # Template injection
        r'../.*/',  # Path traversal
    ]
    
    @staticmethod
    def validate_string_input(value: str, field_name: str) -> ValidationResult:
        """Validate string input for security issues."""
        result = ValidationResult()
        
        if not isinstance(value, str):
            return result
            
        value_lower = value.lower()
        
        for pattern in SecurityValidator.DANGEROUS_PATTERNS:
            if re.search(pattern, value_lower, re.IGNORECASE):
                result.add_issue(
                    field_name,
                    ValidationRule.SECURITY,
                    ValidationSeverity.CRITICAL,
                    f"Potentially dangerous pattern detected: {pattern}",
                    value,
                    "Remove potentially malicious content"
                )
                
        # Check for excessively long strings
        if len(value) > 10000:
            result.add_issue(
                field_name,
                ValidationRule.SECURITY,
                ValidationSeverity.WARNING,
                f"Very long string input: {len(value)} characters",
                suggestion="Consider limiting input length"
            )
            
        return result
        
    @staticmethod 
    def validate_file_path(path: str) -> ValidationResult:
        """Validate file path for security."""
        result = ValidationResult()
        
        # Path traversal check
        if '..' in path:
            result.add_issue(
                "file_path",
                ValidationRule.SECURITY,
                ValidationSeverity.CRITICAL,
                "Path traversal attempt detected",
                path,
                "Use absolute paths or sanitize input"
            )
            
        # Check for suspicious extensions
        dangerous_extensions = ['.exe', '.bat', '.sh', '.ps1', '.cmd']
        for ext in dangerous_extensions:
            if path.lower().endswith(ext):
                result.add_issue(
                    "file_path", 
                    ValidationRule.SECURITY,
                    ValidationSeverity.WARNING,
                    f"Potentially executable file: {ext}",
                    path
                )
                
        return result


class ComprehensiveValidator:
    """Main validation orchestrator."""
    
    def __init__(self):
        self.carbon_validator = CarbonDataValidator()
        self.config_validator = TrainingConfigValidator() 
        self.security_validator = SecurityValidator()
        
    def validate_carbon_data(self, data: Union[CarbonIntensity, CarbonForecast]) -> ValidationResult:
        """Validate carbon data."""
        if isinstance(data, CarbonIntensity):
            return self.carbon_validator.validate_carbon_intensity(data)
        elif isinstance(data, CarbonForecast):
            return self.carbon_validator.validate_forecast(data)
        else:
            result = ValidationResult()
            result.add_issue(
                "data",
                ValidationRule.TYPE_CHECK,
                ValidationSeverity.ERROR,
                f"Unknown carbon data type: {type(data)}"
            )
            return result
            
    def validate_training_config(self, config: TrainingConfig) -> ValidationResult:
        """Validate training configuration."""
        return self.config_validator.validate_config(config)
        
    def validate_api_input(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate API input for security and format."""
        result = ValidationResult()
        
        for key, value in data.items():
            # Security validation for string values
            if isinstance(value, str):
                security_result = self.security_validator.validate_string_input(value, key)
                for issue in security_result.issues:
                    result.issues.append(issue)
                    
            # Check for null bytes
            if isinstance(value, str) and '\x00' in value:
                result.add_issue(
                    key,
                    ValidationRule.SECURITY,
                    ValidationSeverity.CRITICAL,
                    "Null byte detected in input",
                    value,
                    "Remove null bytes from input"
                )
                
        return result
        
    def validate_system_limits(self, **kwargs) -> ValidationResult:
        """Validate system resource limits."""
        result = ValidationResult()
        
        # Memory usage check
        memory_gb = kwargs.get('memory_gb')
        if memory_gb is not None:
            if memory_gb > 100:
                result.add_issue(
                    "memory_gb",
                    ValidationRule.PERFORMANCE,
                    ValidationSeverity.WARNING,
                    f"Very high memory usage: {memory_gb} GB",
                    suggestion="Monitor system resources"
                )
                
        # GPU count check  
        gpu_count = kwargs.get('gpu_count')
        if gpu_count is not None:
            if gpu_count > 16:
                result.add_issue(
                    "gpu_count",
                    ValidationRule.PERFORMANCE,
                    ValidationSeverity.WARNING,
                    f"Very high GPU count: {gpu_count}",
                    suggestion="Ensure adequate power and cooling"
                )
                
        return result
        
    def run_comprehensive_validation(self, **data) -> ValidationResult:
        """Run all applicable validations."""
        result = ValidationResult()
        
        # Security validation for all string inputs
        api_result = self.validate_api_input({k: v for k, v in data.items() if isinstance(v, str)})
        result.issues.extend(api_result.issues)
        
        # System limits validation
        system_result = self.validate_system_limits(**data)
        result.issues.extend(system_result.issues)
        
        # Type-specific validations
        for key, value in data.items():
            if isinstance(value, (CarbonIntensity, CarbonForecast)):
                carbon_result = self.validate_carbon_data(value)
                result.issues.extend(carbon_result.issues)
            elif isinstance(value, TrainingConfig):
                config_result = self.validate_training_config(value)
                result.issues.extend(config_result.issues)
                
        # Update overall validity
        result.is_valid = not result.has_errors()
        
        return result
        
    def get_validation_summary(self, result: ValidationResult) -> Dict[str, Any]:
        """Get human-readable validation summary."""
        issues_by_severity = {}
        for severity in ValidationSeverity:
            issues = result.get_issues_by_severity(severity)
            if issues:
                issues_by_severity[severity.value] = len(issues)
                
        return {
            "is_valid": result.is_valid,
            "total_issues": len(result.issues),
            "issues_by_severity": issues_by_severity,
            "critical_issues": [issue.message for issue in result.get_issues_by_severity(ValidationSeverity.CRITICAL)],
            "error_issues": [issue.message for issue in result.get_issues_by_severity(ValidationSeverity.ERROR)]
        }


# Global validator instance
validator = ComprehensiveValidator()