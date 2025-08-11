"""
Circuit breaker pattern for resilient carbon data provider operations.
"""

import asyncio
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Any, Optional, Dict, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking calls due to failures
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5          # Number of failures before opening
    recovery_timeout: float = 60.0      # Seconds to wait before trying again
    success_threshold: int = 3          # Successful calls needed to close
    timeout: float = 30.0               # Operation timeout in seconds
    monitor_window: int = 300           # Monitoring window in seconds


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    
    def __init__(self, service_name: str, retry_after: float):
        self.service_name = service_name
        self.retry_after = retry_after
        super().__init__(
            f"Circuit breaker OPEN for {service_name}. "
            f"Retry after {retry_after:.1f} seconds."
        )


class CircuitBreaker:
    """Circuit breaker for resilient service calls."""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        
        # Counters
        self.failure_count = 0
        self.success_count = 0
        
        # Timing
        self.last_failure_time: Optional[float] = None
        self.opened_at: Optional[float] = None
        
        # Monitoring
        self.call_history: List[Dict[str, Any]] = []
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        current_time = time.time()
        
        # Check if circuit should transition states
        self._check_state_transition(current_time)
        
        # Block if circuit is open
        if self.state == CircuitState.OPEN:
            retry_after = self._get_retry_after(current_time)
            raise CircuitBreakerError(self.name, retry_after)
            
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )
            
            # Record success
            self._record_success(current_time)
            return result
            
        except asyncio.TimeoutError as e:
            logger.warning(f"Circuit breaker {self.name}: Timeout after {self.config.timeout}s")
            self._record_failure(current_time, "timeout")
            raise
            
        except Exception as e:
            logger.warning(f"Circuit breaker {self.name}: Call failed - {type(e).__name__}: {e}")
            self._record_failure(current_time, str(e))
            raise
            
    def _check_state_transition(self, current_time: float) -> None:
        """Check if circuit breaker should change state."""
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if (self.opened_at and 
                current_time - self.opened_at >= self.config.recovery_timeout):
                logger.info(f"Circuit breaker {self.name}: Transitioning to HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                
        elif self.state == CircuitState.HALF_OPEN:
            # Stay in half-open until success threshold reached
            if self.success_count >= self.config.success_threshold:
                logger.info(f"Circuit breaker {self.name}: Transitioning to CLOSED")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                
    def _record_success(self, current_time: float) -> None:
        """Record successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
        else:
            # Reset failure count on success in closed state
            self.failure_count = 0
            
        self._add_to_history(current_time, "success")
        
    def _record_failure(self, current_time: float, error: str) -> None:
        """Record failed call."""
        self.failure_count += 1
        self.last_failure_time = current_time
        
        # Check if should open circuit
        if (self.state in [CircuitState.CLOSED, CircuitState.HALF_OPEN] and
            self.failure_count >= self.config.failure_threshold):
            logger.warning(
                f"Circuit breaker {self.name}: Opening due to {self.failure_count} failures"
            )
            self.state = CircuitState.OPEN
            self.opened_at = current_time
            
        self._add_to_history(current_time, "failure", error)
        
    def _add_to_history(self, timestamp: float, result: str, error: str = None) -> None:
        """Add call to history for monitoring."""
        entry = {
            "timestamp": timestamp,
            "result": result,
            "state": self.state.value
        }
        if error:
            entry["error"] = error
            
        self.call_history.append(entry)
        
        # Cleanup old entries
        cutoff = timestamp - self.config.monitor_window
        self.call_history = [h for h in self.call_history if h["timestamp"] > cutoff]
        
    def _get_retry_after(self, current_time: float) -> float:
        """Get seconds until circuit can be retried."""
        if not self.opened_at:
            return 0
        elapsed = current_time - self.opened_at
        return max(0, self.config.recovery_timeout - elapsed)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        current_time = time.time()
        cutoff = current_time - self.config.monitor_window
        recent_calls = [h for h in self.call_history if h["timestamp"] > cutoff]
        
        total_calls = len(recent_calls)
        if total_calls == 0:
            success_rate = 1.0
        else:
            successful_calls = len([h for h in recent_calls if h["result"] == "success"])
            success_rate = successful_calls / total_calls
            
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_calls_window": total_calls,
            "success_rate_window": success_rate,
            "last_failure": datetime.fromtimestamp(self.last_failure_time).isoformat() 
                          if self.last_failure_time else None,
            "opened_at": datetime.fromtimestamp(self.opened_at).isoformat()
                        if self.opened_at else None,
            "retry_after": self._get_retry_after(current_time) if self.state == CircuitState.OPEN else 0
        }
        
    def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        logger.info(f"Circuit breaker {self.name}: Manual reset")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.opened_at = None


class CircuitBreakerManager:
    """Manages multiple circuit breakers."""
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
        
    def get_breaker(
        self, 
        name: str, 
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get or create circuit breaker."""
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(name, config)
        return self.breakers[name]
        
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        return {name: breaker.get_stats() for name, breaker in self.breakers.items()}
        
    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self.breakers.values():
            breaker.reset()
            
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary."""
        total_breakers = len(self.breakers)
        if total_breakers == 0:
            return {
                "status": "healthy", 
                "details": "No circuit breakers configured",
                "total_breakers": 0,
                "open_breakers": 0,
                "half_open_breakers": 0,
                "closed_breakers": 0,
                "open_breaker_names": [],
                "half_open_breaker_names": []
            }
            
        open_breakers = [b for b in self.breakers.values() if b.state == CircuitState.OPEN]
        half_open_breakers = [b for b in self.breakers.values() if b.state == CircuitState.HALF_OPEN]
        
        if open_breakers:
            status = "degraded"
        elif half_open_breakers:
            status = "recovering" 
        else:
            status = "healthy"
            
        return {
            "status": status,
            "total_breakers": total_breakers,
            "open_breakers": len(open_breakers),
            "half_open_breakers": len(half_open_breakers),
            "closed_breakers": total_breakers - len(open_breakers) - len(half_open_breakers),
            "open_breaker_names": [b.name for b in open_breakers],
            "half_open_breaker_names": [b.name for b in half_open_breakers]
        }


# Global circuit breaker manager
circuit_breaker_manager = CircuitBreakerManager()