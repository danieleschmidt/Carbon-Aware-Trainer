"""Advanced caching system for carbon-aware trainer."""

import asyncio
import pickle
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from pathlib import Path
import json
import sqlite3
from contextlib import asynccontextmanager
import aiofiles

from .exceptions import CarbonDataError


logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: datetime = None
    size_bytes: int = 0


class CacheStrategy:
    """Cache eviction and management strategies."""
    
    @staticmethod
    def lru_score(entry: CacheEntry) -> float:
        """Least Recently Used scoring."""
        if entry.last_accessed:
            return (datetime.now() - entry.last_accessed).total_seconds()
        return float('inf')
    
    @staticmethod 
    def lfu_score(entry: CacheEntry) -> float:
        """Least Frequently Used scoring."""
        return -entry.access_count  # Negative for ascending sort
    
    @staticmethod
    def ttl_score(entry: CacheEntry) -> float:
        """Time To Live scoring (expires soonest first)."""
        if entry.expires_at:
            return (entry.expires_at - datetime.now()).total_seconds()
        return float('inf')
    
    @staticmethod
    def size_score(entry: CacheEntry) -> float:
        """Size-based scoring (largest first)."""
        return -entry.size_bytes  # Negative for ascending sort


class MemoryCache:
    """High-performance in-memory cache with advanced features."""
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: int = 100,
        default_ttl: Optional[timedelta] = None,
        eviction_strategy: str = "lru"
    ):
        """Initialize memory cache.
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            default_ttl: Default time-to-live for entries
            eviction_strategy: Eviction strategy ('lru', 'lfu', 'ttl', 'size')
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.eviction_strategy = eviction_strategy
        
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        # Strategy function mapping
        self._strategy_functions = {
            'lru': CacheStrategy.lru_score,
            'lfu': CacheStrategy.lfu_score,
            'ttl': CacheStrategy.ttl_score,
            'size': CacheStrategy.size_score
        }
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        async with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                return None
            
            # Check expiration
            if entry.expires_at and datetime.now() > entry.expires_at:
                del self._cache[key]
                self._misses += 1
                return None
            
            # Update access statistics
            entry.access_count += 1
            entry.last_accessed = datetime.now()
            self._hits += 1
            
            return entry.value
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[timedelta] = None
    ) -> bool:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live (uses default if None)
            
        Returns:
            True if value was cached
        """
        async with self._lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except Exception:
                # Fallback size estimation
                size_bytes = len(str(value)) * 4
            
            # Check if single item exceeds memory limit
            if size_bytes > self.max_memory_bytes:
                logger.warning(f"Cache item too large: {size_bytes} bytes")
                return False
            
            # Prepare cache entry
            expires_at = None
            if ttl or self.default_ttl:
                expires_at = datetime.now() + (ttl or self.default_ttl)
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                expires_at=expires_at,
                size_bytes=size_bytes,
                last_accessed=datetime.now()
            )
            
            # Evict if necessary
            await self._evict_if_needed(size_bytes)
            
            # Store entry
            self._cache[key] = entry
            
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete entry from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if entry was deleted
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
    
    async def _evict_if_needed(self, new_item_size: int) -> None:
        """Evict entries if needed to make space.
        
        Args:
            new_item_size: Size of item being added
        """
        # Check size limit
        while len(self._cache) >= self.max_size:
            await self._evict_one()
        
        # Check memory limit
        current_memory = sum(entry.size_bytes for entry in self._cache.values())
        while current_memory + new_item_size > self.max_memory_bytes:
            if not self._cache:  # Safety check
                break
            await self._evict_one()
            current_memory = sum(entry.size_bytes for entry in self._cache.values())
    
    async def _evict_one(self) -> None:
        """Evict one entry based on strategy."""
        if not self._cache:
            return
        
        strategy_func = self._strategy_functions.get(
            self.eviction_strategy, 
            CacheStrategy.lru_score
        )
        
        # Find entry to evict
        entries_with_scores = [
            (key, strategy_func(entry)) 
            for key, entry in self._cache.items()
        ]
        
        # Sort by score (ascending)
        entries_with_scores.sort(key=lambda x: x[1])
        
        # Evict first entry
        key_to_evict = entries_with_scores[0][0]
        del self._cache[key_to_evict]
        self._evictions += 1
        
        logger.debug(f"Evicted cache entry: {key_to_evict}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests) if total_requests > 0 else 0
        
        current_memory = sum(entry.size_bytes for entry in self._cache.values())
        
        return {
            "entries": len(self._cache),
            "max_size": self.max_size,
            "memory_usage_bytes": current_memory,
            "max_memory_bytes": self.max_memory_bytes,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": hit_rate,
            "eviction_strategy": self.eviction_strategy
        }


class PersistentCache:
    """Persistent cache using SQLite for durability."""
    
    def __init__(
        self,
        db_path: Union[str, Path],
        table_name: str = "cache_entries",
        max_size: int = 10000,
        cleanup_interval: timedelta = timedelta(hours=1)
    ):
        """Initialize persistent cache.
        
        Args:
            db_path: Path to SQLite database
            table_name: Table name for cache entries
            max_size: Maximum number of entries
            cleanup_interval: Interval for cleanup operations
        """
        self.db_path = Path(db_path)
        self.table_name = table_name
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval
        
        self._last_cleanup = datetime.now()
        
        # Initialize database
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize SQLite database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    created_at TIMESTAMP,
                    expires_at TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP,
                    size_bytes INTEGER
                )
            """)
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_expires_at ON {self.table_name}(expires_at)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_last_accessed ON {self.table_name}(last_accessed)")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from persistent cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute(f"""
                    SELECT value, expires_at, access_count
                    FROM {self.table_name}
                    WHERE key = ?
                """, (key,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                # Check expiration
                if row['expires_at']:
                    expires_at = datetime.fromisoformat(row['expires_at'])
                    if datetime.now() > expires_at:
                        # Delete expired entry
                        cursor.execute(f"DELETE FROM {self.table_name} WHERE key = ?", (key,))
                        conn.commit()
                        return None
                
                # Update access statistics
                cursor.execute(f"""
                    UPDATE {self.table_name}
                    SET access_count = access_count + 1, last_accessed = ?
                    WHERE key = ?
                """, (datetime.now().isoformat(), key))
                conn.commit()
                
                # Deserialize value
                return pickle.loads(row['value'])
                
        except Exception as e:
            logger.error(f"Error getting from persistent cache: {e}")
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[timedelta] = None
    ) -> bool:
        """Set value in persistent cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live
            
        Returns:
            True if value was cached
        """
        try:
            # Serialize value
            serialized_value = pickle.dumps(value)
            size_bytes = len(serialized_value)
            
            expires_at = None
            if ttl:
                expires_at = (datetime.now() + ttl).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert or replace entry
                cursor.execute(f"""
                    INSERT OR REPLACE INTO {self.table_name}
                    (key, value, created_at, expires_at, access_count, last_accessed, size_bytes)
                    VALUES (?, ?, ?, ?, 0, ?, ?)
                """, (
                    key, 
                    serialized_value, 
                    datetime.now().isoformat(),
                    expires_at,
                    datetime.now().isoformat(),
                    size_bytes
                ))
                conn.commit()
            
            # Cleanup if needed
            await self._cleanup_if_needed()
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting persistent cache: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete entry from persistent cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if entry was deleted
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(f"DELETE FROM {self.table_name} WHERE key = ?", (key,))
                deleted = cursor.rowcount > 0
                conn.commit()
                return deleted
        except Exception as e:
            logger.error(f"Error deleting from persistent cache: {e}")
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(f"DELETE FROM {self.table_name}")
                conn.commit()
        except Exception as e:
            logger.error(f"Error clearing persistent cache: {e}")
    
    async def _cleanup_if_needed(self) -> None:
        """Cleanup expired entries and enforce size limits."""
        now = datetime.now()
        
        if now - self._last_cleanup < self.cleanup_interval:
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Remove expired entries
                cursor.execute(f"""
                    DELETE FROM {self.table_name}
                    WHERE expires_at IS NOT NULL AND expires_at < ?
                """, (now.isoformat(),))
                
                # Enforce size limit (keep most recently accessed)
                cursor.execute(f"""
                    DELETE FROM {self.table_name}
                    WHERE key NOT IN (
                        SELECT key FROM {self.table_name}
                        ORDER BY last_accessed DESC
                        LIMIT ?
                    )
                """, (self.max_size,))
                
                conn.commit()
                
                self._last_cleanup = now
                
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get persistent cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                total_entries = cursor.fetchone()[0]
                
                cursor.execute(f"SELECT SUM(size_bytes) FROM {self.table_name}")
                total_size = cursor.fetchone()[0] or 0
                
                cursor.execute(f"""
                    SELECT COUNT(*) FROM {self.table_name}
                    WHERE expires_at IS NOT NULL AND expires_at < ?
                """, (datetime.now().isoformat(),))
                expired_entries = cursor.fetchone()[0]
                
                return {
                    "entries": total_entries,
                    "max_size": self.max_size,
                    "total_size_bytes": total_size,
                    "expired_entries": expired_entries,
                    "db_path": str(self.db_path)
                }
                
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}


class MultiLevelCache:
    """Multi-level cache combining memory and persistent caches."""
    
    def __init__(
        self,
        memory_cache: Optional[MemoryCache] = None,
        persistent_cache: Optional[PersistentCache] = None,
        write_through: bool = True
    ):
        """Initialize multi-level cache.
        
        Args:
            memory_cache: L1 memory cache
            persistent_cache: L2 persistent cache
            write_through: Whether to write to both levels
        """
        self.memory_cache = memory_cache or MemoryCache()
        self.persistent_cache = persistent_cache
        self.write_through = write_through
        
        # Statistics
        self._l1_hits = 0
        self._l2_hits = 0
        self._misses = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from multi-level cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        # Try L1 cache first
        value = await self.memory_cache.get(key)
        if value is not None:
            self._l1_hits += 1
            return value
        
        # Try L2 cache if available
        if self.persistent_cache:
            value = await self.persistent_cache.get(key)
            if value is not None:
                self._l2_hits += 1
                # Promote to L1 cache
                await self.memory_cache.set(key, value)
                return value
        
        self._misses += 1
        return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[timedelta] = None
    ) -> bool:
        """Set value in multi-level cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live
            
        Returns:
            True if value was cached
        """
        success = True
        
        # Always write to L1
        l1_success = await self.memory_cache.set(key, value, ttl)
        success = success and l1_success
        
        # Write to L2 if write-through enabled
        if self.write_through and self.persistent_cache:
            l2_success = await self.persistent_cache.set(key, value, ttl)
            success = success and l2_success
        
        return success
    
    async def delete(self, key: str) -> bool:
        """Delete entry from all cache levels.
        
        Args:
            key: Cache key
            
        Returns:
            True if entry was deleted from any level
        """
        l1_deleted = await self.memory_cache.delete(key)
        
        l2_deleted = False
        if self.persistent_cache:
            l2_deleted = await self.persistent_cache.delete(key)
        
        return l1_deleted or l2_deleted
    
    async def clear(self) -> None:
        """Clear all cache levels."""
        await self.memory_cache.clear()
        if self.persistent_cache:
            await self.persistent_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get multi-level cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._l1_hits + self._l2_hits + self._misses
        
        stats = {
            "l1_hits": self._l1_hits,
            "l2_hits": self._l2_hits,
            "misses": self._misses,
            "total_requests": total_requests,
            "hit_rate": (self._l1_hits + self._l2_hits) / total_requests if total_requests > 0 else 0,
            "l1_hit_rate": self._l1_hits / total_requests if total_requests > 0 else 0,
            "memory_cache": self.memory_cache.get_stats()
        }
        
        if self.persistent_cache:
            stats["persistent_cache"] = self.persistent_cache.get_stats()
        
        return stats


def cache_key(*args, **kwargs) -> str:
    """Generate cache key from arguments.
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Cache key string
    """
    # Create deterministic string representation
    key_parts = []
    
    for arg in args:
        if hasattr(arg, '__dict__'):
            key_parts.append(str(sorted(arg.__dict__.items())))
        else:
            key_parts.append(str(arg))
    
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}={v}")
    
    key_string = "|".join(key_parts)
    
    # Hash for consistent length
    return hashlib.sha256(key_string.encode()).hexdigest()


def cached(
    cache: Union[MemoryCache, PersistentCache, MultiLevelCache],
    ttl: Optional[timedelta] = None,
    key_func: Optional[Callable] = None
):
    """Decorator for caching function results.
    
    Args:
        cache: Cache instance to use
        ttl: Time-to-live for cached results
        key_func: Custom key generation function
        
    Returns:
        Decorated function
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = f"{func.__name__}:{cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            result = await cache.get(key)
            if result is not None:
                return result
            
            # Execute function and cache result
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            await cache.set(key, result, ttl)
            return result
            
        return wrapper
    return decorator