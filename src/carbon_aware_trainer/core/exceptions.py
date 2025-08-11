"""Custom exceptions for carbon-aware trainer."""

from typing import Optional


class CarbonAwareTrainerError(Exception):
    """Base exception for carbon-aware trainer."""
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code


# Alias for backward compatibility and new modules
CarbonAwareException = CarbonAwareTrainerError


class CarbonDataError(CarbonAwareTrainerError):
    """Raised when carbon data cannot be retrieved or is invalid."""
    pass


class CarbonProviderError(CarbonDataError):
    """Raised when carbon data provider encounters an error."""
    
    def __init__(self, message: str, provider: str, status_code: Optional[int] = None):
        super().__init__(message, error_code=f"PROVIDER_{provider.upper()}_ERROR")
        self.provider = provider
        self.status_code = status_code


class CarbonProviderTimeoutError(CarbonProviderError):
    """Raised when carbon data provider times out."""
    
    def __init__(self, provider: str, timeout_seconds: float):
        super().__init__(
            f"Timeout after {timeout_seconds}s waiting for {provider} response",
            provider=provider
        )
        self.timeout_seconds = timeout_seconds


class CarbonProviderRateLimitError(CarbonProviderError):
    """Raised when carbon data provider rate limit is exceeded."""
    
    def __init__(self, provider: str, retry_after: Optional[int] = None):
        message = f"Rate limit exceeded for {provider}"
        if retry_after:
            message += f". Retry after {retry_after} seconds"
        
        super().__init__(message, provider=provider, status_code=429)
        self.retry_after = retry_after


class TrainingError(CarbonAwareTrainerError):
    """Raised when training encounters an error."""
    pass


class TrainingInterruptedError(TrainingError):
    """Raised when training is interrupted due to carbon constraints."""
    
    def __init__(self, reason: str, carbon_intensity: float, threshold: float):
        super().__init__(
            f"Training interrupted: {reason}. Carbon intensity {carbon_intensity:.1f} > threshold {threshold:.1f} gCO2/kWh",
            error_code="TRAINING_INTERRUPTED"
        )
        self.carbon_intensity = carbon_intensity
        self.threshold = threshold


class ForecastError(CarbonDataError):
    """Raised when carbon forecasting fails."""
    pass


class OptimalWindowNotFoundError(ForecastError):
    """Raised when no optimal training window can be found."""
    
    def __init__(self, region: str, duration_hours: int, max_carbon: float, search_hours: int):
        super().__init__(
            f"No suitable {duration_hours}h training window found in {region} "
            f"within {search_hours}h search window (max carbon: {max_carbon} gCO2/kWh)",
            error_code="NO_OPTIMAL_WINDOW"
        )
        self.region = region
        self.duration_hours = duration_hours
        self.max_carbon = max_carbon
        self.search_hours = search_hours


class ConfigurationError(CarbonAwareTrainerError):
    """Raised when configuration is invalid."""
    pass


class APIKeyError(ConfigurationError):
    """Raised when API key is missing or invalid."""
    
    def __init__(self, provider: str):
        super().__init__(
            f"API key required for {provider}. Set environment variable or pass api_key parameter.",
            error_code=f"MISSING_API_KEY_{provider.upper()}"
        )
        self.provider = provider


class RegionNotSupportedError(ConfigurationError):
    """Raised when region is not supported by provider."""
    
    def __init__(self, region: str, provider: str, supported_regions: Optional[list] = None):
        message = f"Region '{region}' not supported by {provider}"
        if supported_regions:
            message += f". Supported regions: {', '.join(supported_regions[:10])}"
            if len(supported_regions) > 10:
                message += f" (and {len(supported_regions) - 10} more)"
        
        super().__init__(message, error_code=f"UNSUPPORTED_REGION_{provider.upper()}")
        self.region = region
        self.provider = provider
        self.supported_regions = supported_regions


class MonitoringError(CarbonAwareTrainerError):
    """Raised when monitoring encounters an error."""
    pass


class MonitoringNotStartedError(MonitoringError):
    """Raised when monitoring operation requires active monitoring."""
    
    def __init__(self):
        super().__init__(
            "Monitoring not started. Call start_monitoring() first.",
            error_code="MONITORING_NOT_STARTED"
        )


class SchedulingError(CarbonAwareTrainerError):
    """Raised when scheduling encounters an error."""
    pass


class ThresholdExceededError(SchedulingError):
    """Raised when carbon threshold is consistently exceeded."""
    
    def __init__(self, current_intensity: float, threshold: float, duration_minutes: int):
        super().__init__(
            f"Carbon threshold {threshold} gCO2/kWh exceeded for {duration_minutes} minutes "
            f"(current: {current_intensity:.1f} gCO2/kWh)",
            error_code="THRESHOLD_EXCEEDED"
        )
        self.current_intensity = current_intensity
        self.threshold = threshold
        self.duration_minutes = duration_minutes


class MaxPauseDurationExceededError(SchedulingError):
    """Raised when maximum pause duration is exceeded."""
    
    def __init__(self, pause_duration_hours: float, max_duration_hours: float):
        super().__init__(
            f"Training paused for {pause_duration_hours:.1f}h, exceeding maximum {max_duration_hours:.1f}h",
            error_code="MAX_PAUSE_EXCEEDED"
        )
        self.pause_duration_hours = pause_duration_hours
        self.max_duration_hours = max_duration_hours


class MigrationError(CarbonAwareTrainerError):
    """Raised when cross-region migration fails."""
    pass


class MigrationNotAvailableError(MigrationError):
    """Raised when migration is requested but not available."""
    
    def __init__(self, reason: str):
        super().__init__(
            f"Cross-region migration not available: {reason}",
            error_code="MIGRATION_NOT_AVAILABLE"
        )


class CheckpointError(TrainingError):
    """Raised when checkpointing fails."""
    pass


class CheckpointSaveError(CheckpointError):
    """Raised when saving checkpoint fails."""
    
    def __init__(self, checkpoint_path: str, original_error: Exception):
        super().__init__(
            f"Failed to save checkpoint to {checkpoint_path}: {original_error}",
            error_code="CHECKPOINT_SAVE_FAILED"
        )
        self.checkpoint_path = checkpoint_path
        self.original_error = original_error


class CheckpointLoadError(CheckpointError):
    """Raised when loading checkpoint fails."""
    
    def __init__(self, checkpoint_path: str, original_error: Exception):
        super().__init__(
            f"Failed to load checkpoint from {checkpoint_path}: {original_error}",
            error_code="CHECKPOINT_LOAD_FAILED"
        )
        self.checkpoint_path = checkpoint_path
        self.original_error = original_error


class MetricsError(CarbonAwareTrainerError):
    """Raised when metrics collection or reporting fails."""
    pass


class PowerMeteringError(MetricsError):
    """Raised when power consumption measurement fails."""
    
    def __init__(self, device: str, error: Exception):
        super().__init__(
            f"Failed to measure power consumption for {device}: {error}",
            error_code="POWER_METERING_FAILED"
        )
        self.device = device
        self.original_error = error