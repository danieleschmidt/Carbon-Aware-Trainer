"""Retry logic and circuit breaker for robust API calls."""

import asyncio
import random
import time
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Optional, Type, Union, List
from dataclasses import dataclass
from enum import Enum
import functools

from .exceptions import (
    CarbonProviderError, CarbonProviderTimeoutError, 
    CarbonProviderRateLimitError
)


logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Retry strategy types."""
    FIXED = "fixed"
    EXPONENTIAL = "exponential" 
    LINEAR = "linear"
    JITTER = "jitter"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple = (
        ConnectionError, TimeoutError, CarbonProviderTimeoutError
    )


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5      # Failures before opening
    recovery_timeout: float = 60.0  # Seconds before trying half-open
    success_threshold: int = 3      # Successes needed to close from half-open
    timeout: float = 30.0           # Request timeout


class CircuitBreaker:
    """Circuit breaker implementation for service resilience."""
    
    def __init__(self, config: CircuitBreakerConfig):
        """Initialize circuit breaker.
        
        Args:
            config: Circuit breaker configuration
        """
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.next_attempt_time: Optional[datetime] = None
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == CircuitBreakerState.OPEN:
            if (self.next_attempt_time and 
                datetime.now() >= self.next_attempt_time):
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                logger.info("Circuit breaker entering half-open state")
            else:
                raise CarbonProviderError(
                    "Circuit breaker is open - service unavailable",
                    provider="circuit_breaker"
                )
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )
            
            # Handle success
            await self._on_success()
            return result
            
        except Exception as e:
            await self._on_failure(e)
            raise
    
    async def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info("Circuit breaker closed - service recovered")
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0  # Reset failure count on success
    
    async def _on_failure(self, exception: Exception):
        """Handle failed call.
        
        Args:
            exception: Exception that occurred
        """
        logger.warning(f"Circuit breaker recorded failure: {exception}")
        
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if (self.state == CircuitBreakerState.CLOSED and 
            self.failure_count >= self.config.failure_threshold):
            
            self.state = CircuitBreakerState.OPEN
            self.next_attempt_time = (
                datetime.now() + timedelta(seconds=self.config.recovery_timeout)
            )
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures. "
                f"Will retry at {self.next_attempt_time}"
            )
        
        elif self.state == CircuitBreakerState.HALF_OPEN:
            # Failed during half-open, go back to open
            self.state = CircuitBreakerState.OPEN
            self.next_attempt_time = (
                datetime.now() + timedelta(seconds=self.config.recovery_timeout)
            )
            logger.warning("Circuit breaker failed during half-open, reopening")
    
    def get_status(self) -> dict:
        """Get current circuit breaker status.
        
        Returns:
            Dictionary with status information
        """
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
            'next_attempt_time': self.next_attempt_time.isoformat() if self.next_attempt_time else None
        }


class RetryHandler:
    """Advanced retry handler with multiple strategies."""
    
    def __init__(self, config: RetryConfig):
        """Initialize retry handler.
        
        Args:
            config: Retry configuration
        """
        self.config = config
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments  
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: Last exception if all retries exhausted
        """
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                if attempt > 0:
                    logger.info(f"Retry succeeded on attempt {attempt + 1}")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if exception is retryable
                if not self._is_retryable_exception(e):
                    logger.debug(f"Non-retryable exception: {type(e).__name__}")
                    raise
                
                # Don't retry on last attempt
                if attempt == self.config.max_attempts - 1:
                    break
                
                # Calculate delay and wait
                delay = self._calculate_delay(attempt)
                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.2f}s"
                )
                
                await asyncio.sleep(delay)
        
        # All retries exhausted
        logger.error(
            f"All {self.config.max_attempts} retry attempts failed. "
            f"Last error: {last_exception}"
        )
        raise last_exception
    
    def _is_retryable_exception(self, exception: Exception) -> bool:
        """Check if exception should trigger retry.
        
        Args:
            exception: Exception to check
            
        Returns:
            True if should retry
        """
        # Check for rate limiting - special handling
        if isinstance(exception, CarbonProviderRateLimitError):
            return True
        
        # Check configured retryable exceptions
        return isinstance(exception, self.config.retryable_exceptions)
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt.
        
        Args:
            attempt: Current attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        if self.config.strategy == RetryStrategy.FIXED:
            delay = self.config.base_delay
            
        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay * (attempt + 1)
            
        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (
                self.config.backoff_multiplier ** attempt
            )
            
        elif self.config.strategy == RetryStrategy.JITTER:
            # Exponential with jitter
            base_delay = self.config.base_delay * (
                self.config.backoff_multiplier ** attempt
            )
            # Add random jitter (±25%)
            jitter = random.uniform(0.75, 1.25)
            delay = base_delay * jitter
        
        else:
            delay = self.config.base_delay
        
        # Apply maximum delay limit
        return min(delay, self.config.max_delay)


class ResilientCaller:
    """Combines retry logic with circuit breaker for maximum resilience."""
    
    def __init__(
        self, 
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    ):
        """Initialize resilient caller.
        
        Args:
            retry_config: Retry configuration
            circuit_breaker_config: Circuit breaker configuration
        """
        self.retry_handler = RetryHandler(
            retry_config or RetryConfig()
        )
        self.circuit_breaker = CircuitBreaker(
            circuit_breaker_config or CircuitBreakerConfig()
        ) if circuit_breaker_config else None
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with resilience patterns.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        if self.circuit_breaker:
            # Wrap function call with circuit breaker
            async def wrapped_call():
                return await self.retry_handler.execute(func, *args, **kwargs)
            
            return await self.circuit_breaker.call(wrapped_call)
        else:
            # Just use retry handler
            return await self.retry_handler.execute(func, *args, **kwargs)
    
    def get_status(self) -> dict:
        """Get resilience status.
        
        Returns:
            Dictionary with status information
        """
        status = {
            'retry_config': {
                'max_attempts': self.retry_handler.config.max_attempts,
                'strategy': self.retry_handler.config.strategy.value,
                'base_delay': self.retry_handler.config.base_delay,
                'max_delay': self.retry_handler.config.max_delay
            }
        }
        
        if self.circuit_breaker:
            status['circuit_breaker'] = self.circuit_breaker.get_status()
        
        return status


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
):
    """Decorator for adding retry logic to functions.
    
    Args:
        max_attempts: Maximum retry attempts
        base_delay: Base delay between retries
        strategy: Retry strategy to use
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            config = RetryConfig(
                max_attempts=max_attempts,
                base_delay=base_delay,
                strategy=strategy
            )
            handler = RetryHandler(config)
            return await handler.execute(func, *args, **kwargs)
        return wrapper
    return decorator


def with_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0
):
    """Decorator for adding circuit breaker to functions.
    
    Args:
        failure_threshold: Failures before opening circuit
        recovery_timeout: Timeout before trying recovery
        
    Returns:
        Decorated function
    """
    def decorator(func):
        config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )
        circuit_breaker = CircuitBreaker(config)
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await circuit_breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator


# Rate limiting handler
class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, rate: float, burst: int = 1):
        """Initialize rate limiter.
        
        Args:
            rate: Requests per second
            burst: Burst capacity
        """
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last_update = time.time()
    
    async def acquire(self) -> None:
        """Acquire permission to make request.
        
        Raises:
            CarbonProviderRateLimitError: If rate limit exceeded
        """
        now = time.time()
        
        # Add tokens based on elapsed time
        elapsed = now - self.last_update
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_update = now
        
        if self.tokens >= 1:
            self.tokens -= 1
        else:
            # Calculate wait time
            wait_time = (1 - self.tokens) / self.rate
            raise CarbonProviderRateLimitError(
                provider="rate_limiter",
                retry_after=int(wait_time) + 1
            )