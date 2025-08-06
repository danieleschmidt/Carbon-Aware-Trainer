"""Advanced API management with rate limiting, circuit breaking, and fallback."""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import hmac

# Optional dependency handling
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    aiohttp = None
    HAS_AIOHTTP = False

from .retry import ResilientCaller, RateLimiter, CircuitBreaker, RetryConfig, CircuitBreakerConfig
from .exceptions import CarbonProviderError, CarbonProviderTimeoutError, CarbonProviderRateLimitError
from .security import APIKeyManager
from .logging_config import get_performance_logger, get_error_logger
from .metrics_collector import metrics_collector


logger = logging.getLogger(__name__)
performance_logger = get_performance_logger()
error_logger = get_error_logger()


class APIEndpoint(str, Enum):
    """API endpoint types."""
    CURRENT_INTENSITY = "current_intensity"
    FORECAST = "forecast"
    ENERGY_MIX = "energy_mix"
    REGIONS = "regions"


@dataclass
class APICallConfig:
    """Configuration for API calls."""
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_backoff_factor: float = 2.0
    rate_limit_per_minute: float = 60.0
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: float = 60.0
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    enable_compression: bool = True
    user_agent: str = "CarbonAwareTrainer/1.0"


@dataclass
class APICallResult:
    """Result of an API call."""
    success: bool
    data: Optional[Any] = None
    error: Optional[Exception] = None
    duration_ms: float = 0.0
    source: str = "api"
    cached: bool = False
    rate_limited: bool = False
    circuit_breaker_open: bool = False


class APIResponseCache:
    """Thread-safe cache for API responses."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        """Initialize API cache.
        
        Args:
            max_size: Maximum number of cached entries
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cached value.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        async with self._lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # Check if expired
            if datetime.now() > entry['expires_at']:
                del self.cache[key]
                return None
            
            entry['last_accessed'] = datetime.now()
            return entry['data']
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached value.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (uses default if None)
        """
        async with self._lock:
            # Evict old entries if cache is full
            if len(self.cache) >= self.max_size:
                await self._evict_oldest()
            
            ttl = ttl or self.default_ttl
            expires_at = datetime.now() + timedelta(seconds=ttl)
            
            self.cache[key] = {
                'data': value,
                'created_at': datetime.now(),
                'expires_at': expires_at,
                'last_accessed': datetime.now()
            }
    
    async def invalidate(self, pattern: Optional[str] = None) -> int:
        """Invalidate cache entries.
        
        Args:
            pattern: Optional pattern to match keys (simple substring match)
            
        Returns:
            Number of entries invalidated
        """
        async with self._lock:
            if pattern is None:
                # Clear all
                count = len(self.cache)
                self.cache.clear()
                return count
            
            # Remove matching entries
            keys_to_remove = [key for key in self.cache.keys() if pattern in key]
            for key in keys_to_remove:
                del self.cache[key]
            
            return len(keys_to_remove)
    
    async def _evict_oldest(self) -> None:
        """Evict oldest cache entry."""
        if not self.cache:
            return
        
        # Find entry with oldest last_accessed time
        oldest_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k]['last_accessed']
        )
        del self.cache[oldest_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.cache:
            return {
                'size': 0,
                'max_size': self.max_size,
                'hit_rate': 0.0
            }
        
        now = datetime.now()
        expired_count = sum(
            1 for entry in self.cache.values()
            if now > entry['expires_at']
        )
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'expired_entries': expired_count,
            'oldest_entry_age_seconds': (
                now - min(entry['created_at'] for entry in self.cache.values())
            ).total_seconds() if self.cache else 0
        }


class APIManager:
    """Comprehensive API management with resilience patterns."""
    
    def __init__(
        self,
        api_key_manager: Optional[APIKeyManager] = None,
        config: Optional[APICallConfig] = None
    ):
        """Initialize API manager.
        
        Args:
            api_key_manager: API key manager instance
            config: API call configuration
        """
        self.api_key_manager = api_key_manager
        self.config = config or APICallConfig()
        
        # Check if HTTP client is available
        self._http_available = HAS_AIOHTTP
        if not self._http_available:
            logger.warning("aiohttp not available - API calls will fail with helpful errors")
        
        # Rate limiters per provider
        self.rate_limiters: Dict[str, RateLimiter] = {}
        
        # Circuit breakers per provider
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Response cache
        self.cache = APIResponseCache()
        
        # Session for HTTP requests
        self._session: Optional[Any] = None  # Changed from aiohttp.ClientSession
        
        # Statistics tracking
        self.call_stats: Dict[str, Dict[str, int]] = {}
        self.last_stats_reset = datetime.now()
        
        logger.info("APIManager initialized")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
    
    async def start(self) -> None:
        """Start API manager."""
        if not self._http_available:
            logger.info("APIManager started (HTTP functionality disabled - aiohttp not available)")
            return
            
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
            
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={
                    'User-Agent': self.config.user_agent,
                    'Accept': 'application/json',
                    'Accept-Encoding': 'gzip, deflate' if self.config.enable_compression else 'identity'
                }
            )
        
        logger.info("APIManager started")
    
    async def stop(self) -> None:
        """Stop API manager."""
        if self._session:
            await self._session.close()
            self._session = None
        
        logger.info("APIManager stopped")
    
    async def call_api(
        self,
        provider: str,
        endpoint: APIEndpoint,
        url: str,
        method: str = 'GET',
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        cache_key: Optional[str] = None,
        cache_ttl: Optional[int] = None
    ) -> APICallResult:
        """Make API call with full resilience patterns.
        
        Args:
            provider: API provider name
            endpoint: API endpoint type
            url: Request URL
            method: HTTP method
            params: Query parameters
            data: Request body data
            headers: Additional headers
            cache_key: Cache key for response caching
            cache_ttl: Cache TTL override
            
        Returns:
            API call result
        """
        call_start = time.time()
        result = APICallResult(success=False)
        
        try:
            # Check if HTTP functionality is available
            if not self._http_available:
                result.error = CarbonProviderError(
                    "HTTP functionality not available (aiohttp not installed)", 
                    provider
                )
                result.success = False
                return result
            
            # Check cache first
            if cache_key and self.config.enable_caching:
                cached_data = await self.cache.get(cache_key)
                if cached_data is not None:
                    result.success = True
                    result.data = cached_data
                    result.cached = True
                    result.duration_ms = (time.time() - call_start) * 1000
                    
                    self._record_call_stat(provider, endpoint, 'cache_hit')
                    
                    performance_logger.log_api_call(
                        provider=provider,
                        duration_ms=result.duration_ms,
                        success=True,
                        region=params.get('region') if params else None
                    )
                    
                    return result
            
            # Get or create rate limiter
            rate_limiter = await self._get_rate_limiter(provider)
            
            # Get or create circuit breaker
            circuit_breaker = await self._get_circuit_breaker(provider)
            
            # Create resilient caller
            retry_config = RetryConfig(
                max_attempts=self.config.max_retries,
                backoff_multiplier=self.config.retry_backoff_factor
            )
            
            caller = ResilientCaller(
                retry_config=retry_config,
                circuit_breaker_config=None  # Using our own circuit breaker
            )
            
            # Make the actual API call
            async def make_call():
                # Check rate limiting
                try:
                    await rate_limiter.acquire()
                except CarbonProviderRateLimitError as e:
                    result.rate_limited = True
                    raise e
                
                # Check circuit breaker
                if circuit_breaker.state.value == 'open':
                    result.circuit_breaker_open = True
                    raise CarbonProviderError(
                        "Circuit breaker is open",
                        provider=provider
                    )
                
                return await self._make_http_request(
                    provider=provider,
                    url=url,
                    method=method,
                    params=params,
                    data=data,
                    headers=headers
                )
            
            # Execute call with resilience
            try:
                response_data = await caller.call(make_call)
                result.success = True
                result.data = response_data
                
                # Record success for rate limiter and circuit breaker
                rate_limiter.record_success()
                await circuit_breaker._on_success()
                
                # Cache response if configured
                if cache_key and self.config.enable_caching:
                    await self.cache.set(cache_key, response_data, cache_ttl)
                
                self._record_call_stat(provider, endpoint, 'success')
                
            except Exception as e:
                result.error = e
                result.success = False
                
                # Record error for rate limiter and circuit breaker
                rate_limiter.record_error()
                await circuit_breaker._on_failure(e)
                
                self._record_call_stat(provider, endpoint, 'error')
                
                # Log error
                error_logger.log_api_error(
                    provider=provider,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    region=params.get('region') if params else None
                )
                
                raise e
            
        except Exception as e:
            result.error = e
            result.success = False
        
        finally:
            result.duration_ms = (time.time() - call_start) * 1000
            
            # Log performance
            performance_logger.log_api_call(
                provider=provider,
                duration_ms=result.duration_ms,
                success=result.success,
                region=params.get('region') if params else None,
                error=str(result.error) if result.error else None
            )
            
            # Record metrics
            metrics_collector.record_metric(
                f'api.call_duration_ms',
                result.duration_ms,
                {'provider': provider, 'endpoint': endpoint.value, 'success': str(result.success)}
            )
        
        return result
    
    async def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for provider.
        
        Args:
            provider: Provider name
            
        Returns:
            API key or None if not available
        """
        if self.api_key_manager:
            return self.api_key_manager.get_api_key(provider)
        
        # Fallback to environment variables
        import os
        env_var_map = {
            'electricitymap': 'ELECTRICITYMAP_API_KEY',
            'watttime': 'WATTTIME_API_KEY'
        }
        
        env_var = env_var_map.get(provider.lower())
        if env_var:
            return os.getenv(env_var)
        
        return None
    
    async def invalidate_cache(self, provider: Optional[str] = None) -> int:
        """Invalidate cached responses.
        
        Args:
            provider: Optional provider to invalidate (all if None)
            
        Returns:
            Number of entries invalidated
        """
        if provider:
            return await self.cache.invalidate(provider)
        else:
            return await self.cache.invalidate()
    
    def get_call_statistics(self) -> Dict[str, Any]:
        """Get API call statistics.
        
        Returns:
            Dictionary with call statistics
        """
        # Reset stats if it's been more than an hour
        if datetime.now() - self.last_stats_reset > timedelta(hours=1):
            self.call_stats.clear()
            self.last_stats_reset = datetime.now()
        
        # Aggregate statistics
        total_calls = 0
        total_successes = 0
        total_errors = 0
        total_cache_hits = 0
        
        provider_stats = {}
        
        for provider, stats in self.call_stats.items():
            provider_total = sum(stats.values())
            total_calls += provider_total
            total_successes += stats.get('success', 0)
            total_errors += stats.get('error', 0)
            total_cache_hits += stats.get('cache_hit', 0)
            
            provider_stats[provider] = {
                'total_calls': provider_total,
                'success_rate': (stats.get('success', 0) / provider_total * 100) if provider_total > 0 else 0,
                'error_rate': (stats.get('error', 0) / provider_total * 100) if provider_total > 0 else 0,
                'cache_hit_rate': (stats.get('cache_hit', 0) / provider_total * 100) if provider_total > 0 else 0
            }
        
        return {
            'total_calls': total_calls,
            'success_rate': (total_successes / total_calls * 100) if total_calls > 0 else 0,
            'error_rate': (total_errors / total_calls * 100) if total_calls > 0 else 0,
            'cache_hit_rate': (total_cache_hits / total_calls * 100) if total_calls > 0 else 0,
            'provider_stats': provider_stats,
            'cache_stats': self.cache.get_stats(),
            'reset_time': self.last_stats_reset.isoformat()
        }
    
    def get_provider_status(self, provider: str) -> Dict[str, Any]:
        """Get status for specific provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Dictionary with provider status
        """
        status = {
            'provider': provider,
            'has_api_key': False,
            'rate_limiter': None,
            'circuit_breaker': None
        }
        
        # Check API key
        if self.api_key_manager:
            api_key = self.api_key_manager.get_api_key(provider)
            status['has_api_key'] = api_key is not None
        
        # Rate limiter status
        if provider in self.rate_limiters:
            rate_limiter = self.rate_limiters[provider]
            status['rate_limiter'] = rate_limiter.get_status()
        
        # Circuit breaker status
        if provider in self.circuit_breakers:
            circuit_breaker = self.circuit_breakers[provider]
            status['circuit_breaker'] = circuit_breaker.get_status()
        
        return status
    
    async def _get_rate_limiter(self, provider: str) -> RateLimiter:
        """Get or create rate limiter for provider."""
        if provider not in self.rate_limiters:
            self.rate_limiters[provider] = RateLimiter(
                rate=self.config.rate_limit_per_minute / 60.0,  # Convert to per-second
                burst=10,
                adaptive=True
            )
        
        return self.rate_limiters[provider]
    
    async def _get_circuit_breaker(self, provider: str) -> CircuitBreaker:
        """Get or create circuit breaker for provider."""
        if provider not in self.circuit_breakers:
            circuit_config = CircuitBreakerConfig(
                failure_threshold=self.config.circuit_breaker_failure_threshold,
                recovery_timeout=self.config.circuit_breaker_recovery_timeout,
                timeout=self.config.timeout_seconds
            )
            self.circuit_breakers[provider] = CircuitBreaker(circuit_config)
        
        return self.circuit_breakers[provider]
    
    async def _make_http_request(
        self,
        provider: str,
        url: str,
        method: str = 'GET',
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Any:
        """Make HTTP request with proper error handling.
        
        Args:
            provider: Provider name
            url: Request URL
            method: HTTP method
            params: Query parameters
            data: Request body data
            headers: Additional headers
            
        Returns:
            Response data
            
        Raises:
            CarbonProviderError: For API errors
            CarbonProviderTimeoutError: For timeouts
            CarbonProviderRateLimitError: For rate limiting
        """
        if not self._session:
            raise CarbonProviderError("API session not initialized", provider)
        
        # Prepare headers
        request_headers = {}
        if headers:
            request_headers.update(headers)
        
        # Add API key if available
        api_key = await self.get_api_key(provider)
        if api_key:
            # Different providers use different header formats
            if provider.lower() == 'electricitymap':
                request_headers['auth-token'] = api_key
            elif provider.lower() == 'watttime':
                request_headers['Authorization'] = f'Bearer {api_key}'
            else:
                request_headers['Authorization'] = f'Bearer {api_key}'
        
        try:
            async with self._session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=request_headers
            ) as response:
                
                # Handle rate limiting
                if response.status == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    raise CarbonProviderRateLimitError(provider, retry_after)
                
                # Handle authentication errors
                if response.status == 401:
                    raise CarbonProviderError(
                        "Authentication failed - check API key",
                        provider,
                        response.status
                    )
                
                # Handle client errors
                if 400 <= response.status < 500:
                    error_text = await response.text()
                    raise CarbonProviderError(
                        f"Client error: {error_text}",
                        provider,
                        response.status
                    )
                
                # Handle server errors
                if response.status >= 500:
                    error_text = await response.text()
                    raise CarbonProviderError(
                        f"Server error: {error_text}",
                        provider,
                        response.status
                    )
                
                # Success - parse response
                response_data = await response.json()
                return response_data
                
        except Exception as e:
            # Handle aiohttp-specific exceptions only if aiohttp is available
            if HAS_AIOHTTP:
                if isinstance(e, aiohttp.ClientTimeout):
                    raise CarbonProviderTimeoutError(provider, self.config.timeout_seconds)
                elif isinstance(e, aiohttp.ClientConnectionError):
                    raise CarbonProviderError(f"Connection error: {e}", provider)
            
            # Re-raise our own exceptions
            if isinstance(e, (CarbonProviderError, CarbonProviderTimeoutError, CarbonProviderRateLimitError)):
                raise
            raise CarbonProviderError(f"HTTP error: {e}", provider)
    
    def _record_call_stat(self, provider: str, endpoint: APIEndpoint, stat_type: str) -> None:
        """Record call statistics."""
        if provider not in self.call_stats:
            self.call_stats[provider] = {}
        
        key = f"{endpoint.value}_{stat_type}"
        self.call_stats[provider][key] = self.call_stats[provider].get(key, 0) + 1


# Global API manager instance
_api_manager: Optional[APIManager] = None


def get_api_manager(
    api_key_manager: Optional[APIKeyManager] = None,
    config: Optional[APICallConfig] = None
) -> APIManager:
    """Get global API manager instance."""
    global _api_manager
    
    if _api_manager is None:
        _api_manager = APIManager(api_key_manager, config)
    
    return _api_manager