"""Enterprise-scale high-performance caching system for carbon-aware trainer."""

import asyncio
import pickle
import hashlib
import logging
import zlib
import time
import gc
import threading
import weakref
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, OrderedDict
from contextlib import asynccontextmanager

# Optional dependencies
try:
    import lz4.frame as lz4
    HAS_LZ4 = True
except ImportError:
    lz4 = None
    HAS_LZ4 = False

try:
    import redis.asyncio as redis
    HAS_REDIS = True
except ImportError:
    redis = None
    HAS_REDIS = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False


logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with comprehensive metadata for optimization."""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: datetime = None
    size_bytes: int = 0
    compressed: bool = False
    compression_ratio: float = 1.0
    priority: int = 0  # Higher priority = less likely to be evicted
    tags: Set[str] = field(default_factory=set)  # For bulk invalidation
    access_frequency: float = 0.0  # Accesses per minute
    cost_to_recreate: float = 1.0  # Computational cost estimate
    
    def update_access_frequency(self) -> None:
        """Update access frequency based on recent activity."""
        if not self.last_accessed:
            return
        
        time_diff = (datetime.now() - self.last_accessed).total_seconds() / 60  # minutes
        if time_diff > 0:
            self.access_frequency = self.access_count / max(time_diff, 1)


@dataclass
class CacheStats:
    """Comprehensive cache statistics."""
    entries: int = 0
    memory_usage_bytes: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    hit_rate: float = 0.0
    average_entry_size: float = 0.0
    compression_savings_bytes: int = 0
    total_requests: int = 0
    cache_effectiveness: float = 0.0  # How much compute time saved


@dataclass
class CompressedValue:
    """Container for compressed cache values."""
    compressed_data: bytes
    compression_method: str
    original_size: int
    
    def __post_init__(self):
        self.compressed_size = len(self.compressed_data)
        self.compression_ratio = self.original_size / self.compressed_size if self.compressed_size > 0 else 1.0


class CacheStrategy:
    """Advanced cache eviction and management strategies."""
    
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
    
    @staticmethod
    def adaptive_score(entry: CacheEntry) -> float:
        """Adaptive scoring combining multiple factors."""
        entry.update_access_frequency()
        
        # Base score on access frequency
        frequency_score = -entry.access_frequency
        
        # Adjust for priority
        priority_adjustment = entry.priority * 0.1
        
        # Adjust for cost to recreate
        cost_adjustment = -entry.cost_to_recreate * 0.05
        
        # Adjust for size (prefer evicting larger items if similar value)
        size_penalty = entry.size_bytes / 1024 / 1024 * 0.01  # MB penalty
        
        return frequency_score + priority_adjustment + cost_adjustment + size_penalty
    
    @staticmethod
    def w_lru_score(entry: CacheEntry) -> float:
        """Weighted LRU considering entry value and size."""
        if not entry.last_accessed:
            return float('inf')
            
        age_seconds = (datetime.now() - entry.last_accessed).total_seconds()
        size_mb = entry.size_bytes / 1024 / 1024
        access_weight = max(1.0, entry.access_count / 10.0)
        
        # Higher score = more likely to be evicted
        return (age_seconds * size_mb) / access_weight


class HighPerformanceCache:
    """Enterprise-grade high-performance cache with sharding and optimization."""
    
    def __init__(
        self,
        max_size: int = 10000,
        max_memory_mb: int = 500,
        default_ttl: Optional[timedelta] = None,
        eviction_strategy: str = "adaptive",
        compression_threshold: int = 1024,
        enable_compression: bool = True,
        compression_method: str = "lz4",
        enable_weak_refs: bool = True,
        shard_count: int = 16,
        cleanup_interval: int = 60
    ):
        """Initialize high-performance cache.
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            default_ttl: Default time-to-live for entries
            eviction_strategy: Eviction strategy ('lru', 'lfu', 'ttl', 'size', 'adaptive', 'w_lru')
            compression_threshold: Compress values larger than this (bytes)
            enable_compression: Enable value compression
            compression_method: Compression method ('lz4', 'zlib')
            enable_weak_refs: Use weak references for large objects
            shard_count: Number of cache shards for concurrent access
            cleanup_interval: Background cleanup interval in seconds
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.eviction_strategy = eviction_strategy
        self.compression_threshold = compression_threshold
        self.enable_compression = enable_compression and (HAS_LZ4 or compression_method == "zlib")
        self.compression_method = compression_method if HAS_LZ4 else "zlib"
        self.enable_weak_refs = enable_weak_refs
        self.shard_count = shard_count
        self.cleanup_interval = cleanup_interval
        
        # Sharded cache for better concurrency
        self._shards = []
        self._shard_locks = []
        for _ in range(shard_count):
            self._shards.append({})
            self._shard_locks.append(asyncio.Lock())
        
        # Global lock for operations across shards
        self._global_lock = asyncio.Lock()
        
        # Statistics
        self._stats = CacheStats()
        self._compression_stats = {
            'total_compressed': 0,
            'total_compression_savings': 0,
            'compression_time_ms': 0,
            'decompression_time_ms': 0
        }
        
        # Strategy function mapping
        self._strategy_functions = {
            'lru': CacheStrategy.lru_score,
            'lfu': CacheStrategy.lfu_score,
            'ttl': CacheStrategy.ttl_score,
            'size': CacheStrategy.size_score,
            'adaptive': CacheStrategy.adaptive_score,
            'w_lru': CacheStrategy.w_lru_score
        }
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Weak reference cache for large objects
        self._weak_cache: Dict[str, weakref.ref] = {} if enable_weak_refs else None
        
        logger.info(f"High-performance cache initialized (shards: {shard_count}, max_size: {max_size})")
    
    def _get_shard_index(self, key: str) -> int:
        """Get shard index for key using consistent hashing."""
        return hash(key) % self.shard_count
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with optimized sharding."""
        shard_index = self._get_shard_index(key)
        
        async with self._shard_locks[shard_index]:
            entry = self._shards[shard_index].get(key)
            
            if entry is None:
                # Check weak reference cache
                if self._weak_cache and key in self._weak_cache:
                    weak_ref = self._weak_cache[key]
                    value = weak_ref()
                    if value is not None:
                        # Recreate cache entry from weak reference
                        await self._restore_from_weak_ref(key, value)
                        self._stats.hits += 1
                        return value
                    else:
                        del self._weak_cache[key]
                
                self._stats.misses += 1
                return None
            
            # Check expiration
            if entry.expires_at and datetime.now() > entry.expires_at:
                del self._shards[shard_index][key]
                self._stats.misses += 1
                return None
            
            # Update access statistics
            entry.access_count += 1
            entry.last_accessed = datetime.now()
            self._stats.hits += 1
            
            # Decompress if needed
            value = await self._decompress_value(entry.value)
            return value
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[timedelta] = None,
        priority: int = 0,
        tags: Optional[Set[str]] = None,
        cost_to_recreate: float = 1.0
    ) -> bool:
        """Set value in cache with advanced options."""
        shard_index = self._get_shard_index(key)
        
        # Calculate size and compress if needed
        processed_value, size_bytes, compressed = await self._prepare_value_for_storage(value)
        
        # Check if single item exceeds memory limit
        if size_bytes > self.max_memory_bytes:
            logger.warning(f"Cache item too large: {size_bytes} bytes")
            return False
        
        async with self._shard_locks[shard_index]:
            # Prepare cache entry
            expires_at = None
            if ttl or self.default_ttl:
                expires_at = datetime.now() + (ttl or self.default_ttl)
            
            entry = CacheEntry(
                key=key,
                value=processed_value,
                created_at=datetime.now(),
                expires_at=expires_at,
                size_bytes=size_bytes,
                last_accessed=datetime.now(),
                compressed=compressed,
                priority=priority,
                tags=tags or set(),
                cost_to_recreate=cost_to_recreate
            )
            
            # Store entry
            self._shards[shard_index][key] = entry
        
        # Global eviction check
        await self._global_eviction_check()
        
        # Store weak reference for large objects
        if self._weak_cache and size_bytes > 1024 * 1024:  # > 1MB
            await self._store_weak_reference(key, value)
        
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        shard_index = self._get_shard_index(key)
        
        async with self._shard_locks[shard_index]:
            if key in self._shards[shard_index]:
                del self._shards[shard_index][key]
                
                # Remove weak reference if exists
                if self._weak_cache and key in self._weak_cache:
                    del self._weak_cache[key]
                
                return True
            return False
    
    async def delete_by_tags(self, tags: Set[str]) -> int:
        """Delete entries by tags."""
        deleted_count = 0
        
        for shard_idx in range(self.shard_count):
            async with self._shard_locks[shard_idx]:
                keys_to_delete = []
                
                for key, entry in self._shards[shard_idx].items():
                    if entry.tags.intersection(tags):
                        keys_to_delete.append(key)
                
                for key in keys_to_delete:
                    del self._shards[shard_idx][key]
                    deleted_count += 1
                    
                    # Remove weak reference if exists
                    if self._weak_cache and key in self._weak_cache:
                        del self._weak_cache[key]
        
        return deleted_count
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        for shard_idx in range(self.shard_count):
            async with self._shard_locks[shard_idx]:
                self._shards[shard_idx].clear()
        
        if self._weak_cache:
            self._weak_cache.clear()
        
        # Reset statistics
        self._stats = CacheStats()
        self._compression_stats = {
            'total_compressed': 0,
            'total_compression_savings': 0,
            'compression_time_ms': 0,
            'decompression_time_ms': 0
        }
    
    async def start_background_cleanup(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task and not self._cleanup_task.done():
            return
        
        self._cleanup_task = asyncio.create_task(self._background_cleanup())
        logger.info("Started cache background cleanup task")
    
    async def stop_background_cleanup(self) -> None:
        """Stop background cleanup task."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped cache background cleanup task")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        total_requests = self._stats.hits + self._stats.misses
        hit_rate = (self._stats.hits / total_requests) if total_requests > 0 else 0
        
        # Calculate memory usage across all shards
        current_memory = 0
        total_entries = 0
        compressed_entries = 0
        
        for shard in self._shards:
            for entry in shard.values():
                current_memory += entry.size_bytes
                total_entries += 1
                if entry.compressed:
                    compressed_entries += 1
        
        avg_entry_size = current_memory / total_entries if total_entries > 0 else 0
        compression_ratio = compressed_entries / total_entries if total_entries > 0 else 0
        
        return {
            "entries": total_entries,
            "max_size": self.max_size,
            "memory_usage_bytes": current_memory,
            "max_memory_bytes": self.max_memory_bytes,
            "memory_utilization_pct": (current_memory / self.max_memory_bytes * 100) if self.max_memory_bytes > 0 else 0,
            "hits": self._stats.hits,
            "misses": self._stats.misses,
            "evictions": self._stats.evictions,
            "hit_rate": hit_rate,
            "average_entry_size_bytes": avg_entry_size,
            "compression_enabled": self.enable_compression,
            "compressed_entries_ratio": compression_ratio,
            "compression_savings_bytes": self._compression_stats['total_compression_savings'],
            "eviction_strategy": self.eviction_strategy,
            "shard_count": self.shard_count,
            "weak_refs_count": len(self._weak_cache) if self._weak_cache else 0,
            "compression_stats": self._compression_stats.copy()
        }
    
    async def _prepare_value_for_storage(self, value: Any) -> Tuple[Any, int, bool]:
        """Prepare value for storage (compression, serialization)."""
        start_time = time.time()
        
        try:
            # Serialize value
            serialized_data = pickle.dumps(value)
            original_size = len(serialized_data)
            
            # Compress if enabled and above threshold
            if (self.enable_compression and 
                original_size > self.compression_threshold):
                
                if self.compression_method == 'lz4' and HAS_LZ4:
                    compressed_data = lz4.compress(serialized_data)
                elif self.compression_method == 'zlib':
                    compressed_data = zlib.compress(serialized_data)
                else:
                    compressed_data = serialized_data  # No compression
                
                if len(compressed_data) < original_size * 0.9:  # Only use if >10% savings
                    compressed_value = CompressedValue(
                        compressed_data, self.compression_method, original_size
                    )
                    
                    compression_time = (time.time() - start_time) * 1000
                    self._compression_stats['compression_time_ms'] += compression_time
                    self._compression_stats['total_compressed'] += 1
                    self._compression_stats['total_compression_savings'] += (original_size - len(compressed_data))
                    
                    return compressed_value, len(compressed_data), True
            
            return value, original_size, False
            
        except Exception as e:
            logger.warning(f"Value preparation failed: {e}")
            # Fallback to storing raw value
            size_estimate = len(str(value)) * 4
            return value, size_estimate, False
    
    async def _decompress_value(self, value: Any) -> Any:
        """Decompress value if it's compressed."""
        if not isinstance(value, CompressedValue):
            return value
        
        start_time = time.time()
        
        try:
            if value.compression_method == 'lz4' and HAS_LZ4:
                decompressed_data = lz4.decompress(value.compressed_data)
            elif value.compression_method == 'zlib':
                decompressed_data = zlib.decompress(value.compressed_data)
            else:
                return pickle.loads(value.compressed_data)  # Fallback
            
            result = pickle.loads(decompressed_data)
            
            decompression_time = (time.time() - start_time) * 1000
            self._compression_stats['decompression_time_ms'] += decompression_time
            
            return result
            
        except Exception as e:
            logger.warning(f"Decompression failed: {e}")
            return None
    
    async def _global_eviction_check(self) -> None:
        """Global eviction check across all shards."""
        async with self._global_lock:
            # Count total entries and memory
            total_entries = sum(len(shard) for shard in self._shards)
            total_memory = sum(
                sum(entry.size_bytes for entry in shard.values())
                for shard in self._shards
            )
            
            # Evict if needed
            while total_entries > self.max_size or total_memory > self.max_memory_bytes:
                evicted = await self._evict_globally()
                if not evicted:
                    break  # No more entries to evict
                
                total_entries -= 1
                total_memory = sum(
                    sum(entry.size_bytes for entry in shard.values())
                    for shard in self._shards
                )
    
    async def _evict_globally(self) -> bool:
        """Evict one entry globally using the configured strategy."""
        # Collect candidates from all shards
        candidates = []
        
        for shard_idx, shard in enumerate(self._shards):
            for key, entry in shard.items():
                candidates.append((shard_idx, key, entry))
        
        if not candidates:
            return False
        
        # Score all candidates
        strategy_func = self._strategy_functions.get(
            self.eviction_strategy,
            CacheStrategy.adaptive_score
        )
        
        scored_candidates = [
            (shard_idx, key, strategy_func(entry))
            for shard_idx, key, entry in candidates
        ]
        
        # Sort by score and evict the worst
        scored_candidates.sort(key=lambda x: x[2])
        
        shard_idx, key_to_evict, _ = scored_candidates[0]
        
        async with self._shard_locks[shard_idx]:
            if key_to_evict in self._shards[shard_idx]:
                del self._shards[shard_idx][key_to_evict]
                self._stats.evictions += 1
                logger.debug(f"Globally evicted cache entry: {key_to_evict}")
                return True
        
        return False
    
    async def _store_weak_reference(self, key: str, value: Any) -> None:
        """Store weak reference for large objects."""
        if not self._weak_cache:
            return
        
        try:
            self._weak_cache[key] = weakref.ref(value)
        except TypeError:
            # Some objects don't support weak references
            pass
    
    async def _restore_from_weak_ref(self, key: str, value: Any) -> None:
        """Restore cache entry from weak reference."""
        shard_index = self._get_shard_index(key)
        
        # Re-cache the value with low priority
        processed_value, size_bytes, compressed = await self._prepare_value_for_storage(value)
        
        entry = CacheEntry(
            key=key,
            value=processed_value,
            created_at=datetime.now(),
            size_bytes=size_bytes,
            last_accessed=datetime.now(),
            compressed=compressed,
            priority=-1  # Low priority for restored entries
        )
        
        self._shards[shard_index][key] = entry
    
    async def _background_cleanup(self) -> None:
        """Background task for cleanup operations."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                # Clean expired entries
                await self._cleanup_expired_entries()
                
                # Trigger garbage collection periodically
                if self._stats.entries > 1000:
                    gc.collect()
                
                # Clean up weak references
                await self._cleanup_weak_references()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background cleanup error: {e}")
    
    async def _cleanup_expired_entries(self) -> None:
        """Clean up expired entries across all shards."""
        now = datetime.now()
        cleaned_count = 0
        
        for shard_idx in range(self.shard_count):
            async with self._shard_locks[shard_idx]:
                keys_to_delete = []
                
                for key, entry in self._shards[shard_idx].items():
                    if entry.expires_at and now > entry.expires_at:
                        keys_to_delete.append(key)
                
                for key in keys_to_delete:
                    del self._shards[shard_idx][key]
                    cleaned_count += 1
                    
                    # Remove weak reference if exists
                    if self._weak_cache and key in self._weak_cache:
                        del self._weak_cache[key]
        
        if cleaned_count > 0:
            logger.debug(f"Cleaned up {cleaned_count} expired cache entries")
    
    async def _cleanup_weak_references(self) -> None:
        """Clean up dead weak references."""
        if not self._weak_cache:
            return
        
        dead_refs = []
        for key, weak_ref in self._weak_cache.items():
            if weak_ref() is None:
                dead_refs.append(key)
        
        for key in dead_refs:
            del self._weak_cache[key]
        
        if dead_refs:
            logger.debug(f"Cleaned up {len(dead_refs)} dead weak references")


class RedisCache:
    """Distributed Redis cache for multi-instance deployments."""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "carbon_cache:",
        default_ttl: int = 300,
        serialization_method: str = "pickle",
        compression_enabled: bool = True,
        max_connections: int = 10
    ):
        """Initialize Redis cache."""
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.default_ttl = default_ttl
        self.serialization_method = serialization_method
        self.compression_enabled = compression_enabled and HAS_LZ4
        self.max_connections = max_connections
        
        self._redis: Optional[redis.Redis] = None
        self._connection_pool: Optional[redis.ConnectionPool] = None
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0
        }
        
        if not HAS_REDIS:
            logger.warning("Redis not available - distributed caching disabled")
    
    async def start(self) -> None:
        """Start Redis connection."""
        if not HAS_REDIS:
            return
        
        try:
            self._connection_pool = redis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.max_connections
            )
            self._redis = redis.Redis(connection_pool=self._connection_pool)
            await self._redis.ping()
            logger.info("Redis cache connected")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._redis = None
    
    async def stop(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
        
        if self._connection_pool:
            await self._connection_pool.disconnect()
            self._connection_pool = None
        
        logger.info("Redis cache disconnected")
    
    def _make_key(self, key: str) -> str:
        """Create full Redis key with prefix."""
        return f"{self.key_prefix}{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if not self._redis:
            return None
        
        try:
            full_key = self._make_key(key)
            data = await self._redis.get(full_key)
            
            if data is None:
                self._stats['misses'] += 1
                return None
            
            # Deserialize value
            value = self._deserialize(data)
            self._stats['hits'] += 1
            return value
            
        except Exception as e:
            logger.warning(f"Redis get error: {e}")
            self._stats['errors'] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache."""
        if not self._redis:
            return False
        
        try:
            full_key = self._make_key(key)
            serialized_value = self._serialize(value)
            ttl = ttl or self.default_ttl
            
            await self._redis.setex(full_key, ttl, serialized_value)
            self._stats['sets'] += 1
            return True
            
        except Exception as e:
            logger.warning(f"Redis set error: {e}")
            self._stats['errors'] += 1
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis cache."""
        if not self._redis:
            return False
        
        try:
            full_key = self._make_key(key)
            deleted = await self._redis.delete(full_key)
            self._stats['deletes'] += 1
            return deleted > 0
            
        except Exception as e:
            logger.warning(f"Redis delete error: {e}")
            self._stats['errors'] += 1
            return False
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for Redis storage."""
        data = pickle.dumps(value)
        
        if self.compression_enabled and len(data) > 1024:
            try:
                compressed = lz4.compress(data)
                if len(compressed) < len(data):
                    return b'lz4:' + compressed
            except Exception:
                pass
        
        return data
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from Redis storage."""
        # Check for compression marker
        if data.startswith(b'lz4:'):
            try:
                data = lz4.decompress(data[4:])
            except Exception as e:
                logger.warning(f"LZ4 decompression failed: {e}")
                raise
        
        return pickle.loads(data)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics."""
        total_ops = sum(self._stats.values())
        hit_rate = self._stats['hits'] / (self._stats['hits'] + self._stats['misses']) if (self._stats['hits'] + self._stats['misses']) > 0 else 0
        
        return {
            'connected': self._redis is not None,
            'hit_rate': hit_rate,
            'total_operations': total_ops,
            **self._stats
        }


class ConnectionPool:
    """Advanced HTTP connection pool with intelligent management."""
    
    def __init__(
        self,
        max_connections: int = 100,
        max_connections_per_host: int = 10,
        connection_timeout: float = 30.0,
        read_timeout: float = 60.0,
        keepalive_timeout: float = 120.0,
        enable_http2: bool = False
    ):
        """Initialize connection pool."""
        self.max_connections = max_connections
        self.max_connections_per_host = max_connections_per_host
        self.connection_timeout = connection_timeout
        self.read_timeout = read_timeout
        self.keepalive_timeout = keepalive_timeout
        self.enable_http2 = enable_http2
        
        self._session: Optional[Any] = None  # aiohttp.ClientSession
        self._connector: Optional[Any] = None  # aiohttp.TCPConnector
        
        # Connection statistics
        self._stats = {
            'total_requests': 0,
            'active_connections': 0,
            'failed_connections': 0,
            'timeouts': 0,
            'reused_connections': 0
        }
    
    async def start(self) -> None:
        """Initialize connection pool."""
        try:
            import aiohttp
            
            # Create connector with connection pooling settings
            self._connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=self.max_connections_per_host,
                keepalive_timeout=self.keepalive_timeout,
                enable_cleanup_closed=True,
                use_dns_cache=True,
                ttl_dns_cache=300,  # 5 minutes DNS cache
                family=0,  # Allow both IPv4 and IPv6
            )
            
            # Create session with timeouts
            timeout = aiohttp.ClientTimeout(
                total=self.connection_timeout + self.read_timeout,
                connect=self.connection_timeout,
                sock_read=self.read_timeout
            )
            
            self._session = aiohttp.ClientSession(
                connector=self._connector,
                timeout=timeout,
                headers={
                    'User-Agent': 'CarbonAwareTrainer/1.0',
                    'Connection': 'keep-alive'
                }
            )
            
            logger.info(f"HTTP connection pool started (max: {self.max_connections})")
            
        except ImportError:
            logger.warning("aiohttp not available - HTTP connection pooling disabled")
    
    async def stop(self) -> None:
        """Close connection pool."""
        if self._session:
            await self._session.close()
            self._session = None
        
        if self._connector:
            await self._connector.close()
            self._connector = None
        
        logger.info("HTTP connection pool stopped")
    
    @property
    def session(self) -> Optional[Any]:
        """Get HTTP session for making requests."""
        return self._session
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        stats = self._stats.copy()
        
        if self._connector:
            # Add connector-specific stats if available
            try:
                stats['pool_size'] = len(self._connector._conns)
                stats['acquired_connections'] = len(self._connector._acquired)
            except AttributeError:
                pass
        
        return stats


class MemoryManager:
    """Memory management and optimization utilities."""
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage information."""
        if not HAS_PSUTIL:
            return {'error': 'psutil not available'}
        
        import os
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    
    @staticmethod
    def optimize_memory() -> Dict[str, int]:
        """Perform memory optimization."""
        # Force garbage collection
        collected = {
            'gen0': gc.collect(0),
            'gen1': gc.collect(1),
            'gen2': gc.collect(2)
        }
        
        # Get garbage collection stats
        stats = gc.get_stats()
        
        return {
            'collected_objects': sum(collected.values()),
            'gc_stats': stats,
            **collected
        }
    
    @staticmethod
    def detect_memory_leaks() -> List[Dict[str, Any]]:
        """Detect potential memory leaks."""
        from collections import Counter
        
        # Get all objects in memory
        all_objects = gc.get_objects()
        
        # Count by type
        type_counts = Counter(type(obj).__name__ for obj in all_objects)
        
        # Look for suspicious patterns
        suspicious_types = []
        for type_name, count in type_counts.most_common(20):
            if count > 1000:  # Arbitrary threshold
                suspicious_types.append({
                    'type': type_name,
                    'count': count,
                    'size_estimate': count * 64  # Rough estimate
                })
        
        return suspicious_types


# Global cache instance
_global_cache: Optional[HighPerformanceCache] = None
_global_redis_cache: Optional[RedisCache] = None
_global_connection_pool: Optional[ConnectionPool] = None


def get_global_cache(
    memory_cache_config: Optional[Dict[str, Any]] = None,
    redis_config: Optional[Dict[str, Any]] = None
) -> HighPerformanceCache:
    """Get global high-performance cache instance."""
    global _global_cache
    
    if _global_cache is None:
        config = memory_cache_config or {}
        _global_cache = HighPerformanceCache(**config)
        
        # Start background cleanup
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(_global_cache.start_background_cleanup())
        except RuntimeError:
            # No event loop running
            pass
    
    return _global_cache


def get_global_redis_cache(config: Optional[Dict[str, Any]] = None) -> RedisCache:
    """Get global Redis cache instance."""
    global _global_redis_cache
    
    if _global_redis_cache is None:
        redis_config = config or {}
        _global_redis_cache = RedisCache(**redis_config)
    
    return _global_redis_cache


def get_global_connection_pool(config: Optional[Dict[str, Any]] = None) -> ConnectionPool:
    """Get global HTTP connection pool."""
    global _global_connection_pool
    
    if _global_connection_pool is None:
        pool_config = config or {}
        _global_connection_pool = ConnectionPool(**pool_config)
    
    return _global_connection_pool


# Cache decorators for easy usage
def cached(
    cache: Optional[HighPerformanceCache] = None,
    ttl: Optional[timedelta] = None,
    key_func: Optional[Callable] = None,
    priority: int = 0,
    tags: Optional[Set[str]] = None
):
    """Decorator for caching function results."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cache_instance = cache or get_global_cache()
            
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key_data = f"{func.__name__}:{args}:{sorted(kwargs.items())}"
                key = hashlib.sha256(key_data.encode()).hexdigest()
            
            # Try to get from cache
            result = await cache_instance.get(key)
            if result is not None:
                return result
            
            # Execute function and cache result
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            await cache_instance.set(key, result, ttl, priority, tags)
            return result
            
        return wrapper
    return decorator