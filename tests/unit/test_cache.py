"""Unit tests for caching system."""

import pytest
import asyncio
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from carbon_aware_trainer.core.cache import (
    MemoryCache, PersistentCache, MultiLevelCache,
    CacheStrategy, CacheEntry, cache_key, cached
)


class TestMemoryCache:
    """Test cases for MemoryCache."""
    
    @pytest.fixture
    def cache(self):
        """Create memory cache for testing."""
        return MemoryCache(max_size=10, max_memory_mb=1, default_ttl=timedelta(seconds=1))
    
    @pytest.mark.asyncio
    async def test_basic_get_set(self, cache):
        """Test basic cache get/set operations."""
        # Test set and get
        assert await cache.set("key1", "value1")
        assert await cache.get("key1") == "value1"
        
        # Test non-existent key
        assert await cache.get("nonexistent") is None
    
    @pytest.mark.asyncio
    async def test_ttl_expiration(self, cache):
        """Test TTL-based expiration."""
        # Set with short TTL
        await cache.set("key1", "value1", ttl=timedelta(milliseconds=100))
        
        # Should be available immediately
        assert await cache.get("key1") == "value1"
        
        # Wait for expiration
        await asyncio.sleep(0.2)
        
        # Should be expired
        assert await cache.get("key1") is None
    
    @pytest.mark.asyncio
    async def test_size_eviction(self, cache):
        """Test size-based eviction."""
        # Fill cache to capacity
        for i in range(10):
            await cache.set(f"key{i}", f"value{i}")
        
        # Add one more item (should evict least recently used)
        await cache.set("key10", "value10")
        
        # Check that we have exactly max_size items
        stats = cache.get_stats()
        assert stats["entries"] == 10
        assert stats["evictions"] > 0
    
    @pytest.mark.asyncio
    async def test_delete_and_clear(self, cache):
        """Test delete and clear operations."""
        # Add some items
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        
        # Test delete
        assert await cache.delete("key1") is True
        assert await cache.get("key1") is None
        assert await cache.get("key2") == "value2"
        
        # Test clear
        await cache.clear()
        assert await cache.get("key2") is None
        
        stats = cache.get_stats()
        assert stats["entries"] == 0
    
    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """Test LRU eviction strategy."""
        cache = MemoryCache(max_size=3, eviction_strategy="lru")
        
        # Add items
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")
        
        # Access key1 to make it recently used
        await cache.get("key1")
        
        # Add new item (should evict key2 as least recently used)
        await cache.set("key4", "value4")
        
        # key1 and key3, key4 should exist, key2 should be evicted
        assert await cache.get("key1") == "value1"
        assert await cache.get("key2") is None
        assert await cache.get("key3") == "value3"
        assert await cache.get("key4") == "value4"
    
    def test_cache_stats(self, cache):
        """Test cache statistics."""
        stats = cache.get_stats()
        
        assert "entries" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert stats["eviction_strategy"] == "lru"


class TestPersistentCache:
    """Test cases for PersistentCache."""
    
    @pytest.fixture
    def cache(self):
        """Create persistent cache for testing."""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test_cache.db"
        return PersistentCache(db_path, max_size=100)
    
    @pytest.mark.asyncio
    async def test_persistent_get_set(self, cache):
        """Test persistent cache operations."""
        # Test set and get
        assert await cache.set("key1", {"data": "value1"})
        result = await cache.get("key1")
        assert result == {"data": "value1"}
    
    @pytest.mark.asyncio
    async def test_persistent_ttl(self, cache):
        """Test TTL in persistent cache."""
        # Set with short TTL
        await cache.set("key1", "value1", ttl=timedelta(milliseconds=100))
        
        # Should be available immediately
        assert await cache.get("key1") == "value1"
        
        # Wait for expiration
        await asyncio.sleep(0.2)
        
        # Should be expired and auto-deleted
        assert await cache.get("key1") is None
    
    @pytest.mark.asyncio
    async def test_persistent_stats(self, cache):
        """Test persistent cache statistics."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        
        stats = cache.get_stats()
        assert stats["entries"] == 2
        assert "total_size_bytes" in stats
        assert "expired_entries" in stats


class TestMultiLevelCache:
    """Test cases for MultiLevelCache."""
    
    @pytest.fixture
    def cache(self):
        """Create multi-level cache for testing."""
        memory_cache = MemoryCache(max_size=5)
        
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test_multi_cache.db"
        persistent_cache = PersistentCache(db_path)
        
        return MultiLevelCache(memory_cache, persistent_cache)
    
    @pytest.mark.asyncio
    async def test_multi_level_promotion(self, cache):
        """Test promotion from L2 to L1 cache."""
        # Set in L2 only (by bypassing L1)
        await cache.persistent_cache.set("key1", "value1")
        
        # Get should promote to L1
        result = await cache.get("key1")
        assert result == "value1"
        
        # Should now be in L1 cache
        l1_result = await cache.memory_cache.get("key1")
        assert l1_result == "value1"
    
    @pytest.mark.asyncio
    async def test_multi_level_write_through(self, cache):
        """Test write-through behavior."""
        # Set should write to both levels
        await cache.set("key1", "value1")
        
        # Should be in both caches
        assert await cache.memory_cache.get("key1") == "value1"
        assert await cache.persistent_cache.get("key1") == "value1"
    
    @pytest.mark.asyncio
    async def test_multi_level_stats(self, cache):
        """Test multi-level cache statistics."""
        # Add some data
        await cache.set("key1", "value1")
        await cache.get("key1")  # L1 hit
        
        # Clear L1, access again
        await cache.memory_cache.clear()
        await cache.get("key1")  # L2 hit
        
        stats = cache.get_stats()
        assert stats["l1_hits"] > 0
        assert stats["l2_hits"] > 0
        assert "hit_rate" in stats


class TestCacheUtilities:
    """Test cache utility functions."""
    
    def test_cache_key_generation(self):
        """Test cache key generation."""
        # Test with simple arguments
        key1 = cache_key("arg1", "arg2", kwarg1="value1")
        key2 = cache_key("arg1", "arg2", kwarg1="value1")
        key3 = cache_key("arg1", "arg2", kwarg1="value2")
        
        assert key1 == key2  # Same arguments should produce same key
        assert key1 != key3  # Different arguments should produce different key
        assert len(key1) == 64  # SHA256 hash length
    
    def test_cache_key_with_objects(self):
        """Test cache key generation with objects."""
        class TestObj:
            def __init__(self, value):
                self.value = value
        
        obj1 = TestObj("test")
        obj2 = TestObj("test")
        
        key1 = cache_key(obj1)
        key2 = cache_key(obj2)
        
        assert key1 == key2  # Objects with same attributes should produce same key
    
    @pytest.mark.asyncio
    async def test_cached_decorator(self):
        """Test cached decorator functionality."""
        cache = MemoryCache()
        call_count = 0
        
        @cached(cache, ttl=timedelta(seconds=1))
        async def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call should execute function
        result1 = await expensive_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call should use cache
        result2 = await expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Function not called again
        
        # Different argument should execute function
        result3 = await expensive_function(10)
        assert result3 == 20
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_cached_decorator_with_custom_key(self):
        """Test cached decorator with custom key function."""
        cache = MemoryCache()
        
        def custom_key_func(x, y):
            return f"custom_{x}_{y}"
        
        @cached(cache, key_func=custom_key_func)
        async def test_function(x, y):
            return x + y
        
        result = await test_function(1, 2)
        assert result == 3
        
        # Check that custom key was used
        cached_result = await cache.get("custom_1_2")
        assert cached_result == 3


class TestCacheStrategies:
    """Test cache eviction strategies."""
    
    def test_lru_scoring(self):
        """Test LRU scoring strategy."""
        now = datetime.now()
        
        # Recent entry should have low score (less likely to evict)
        recent_entry = CacheEntry(
            key="recent",
            value="value",
            created_at=now,
            last_accessed=now
        )
        
        # Old entry should have high score (more likely to evict)
        old_entry = CacheEntry(
            key="old",
            value="value", 
            created_at=now - timedelta(hours=1),
            last_accessed=now - timedelta(hours=1)
        )
        
        recent_score = CacheStrategy.lru_score(recent_entry)
        old_score = CacheStrategy.lru_score(old_entry)
        
        assert recent_score < old_score
    
    def test_lfu_scoring(self):
        """Test LFU scoring strategy."""
        frequent_entry = CacheEntry(
            key="frequent",
            value="value",
            created_at=datetime.now(),
            access_count=100
        )
        
        infrequent_entry = CacheEntry(
            key="infrequent", 
            value="value",
            created_at=datetime.now(),
            access_count=1
        )
        
        frequent_score = CacheStrategy.lfu_score(frequent_entry)
        infrequent_score = CacheStrategy.lfu_score(infrequent_entry)
        
        # LFU uses negative count, so frequent should have lower (more negative) score
        assert frequent_score < infrequent_score
    
    def test_ttl_scoring(self):
        """Test TTL scoring strategy."""
        now = datetime.now()
        
        soon_expire = CacheEntry(
            key="soon",
            value="value",
            created_at=now,
            expires_at=now + timedelta(minutes=1)
        )
        
        later_expire = CacheEntry(
            key="later",
            value="value",
            created_at=now,
            expires_at=now + timedelta(hours=1)
        )
        
        soon_score = CacheStrategy.ttl_score(soon_expire)
        later_score = CacheStrategy.ttl_score(later_expire)
        
        assert soon_score < later_score
    
    def test_size_scoring(self):
        """Test size-based scoring strategy."""
        large_entry = CacheEntry(
            key="large",
            value="value",
            created_at=datetime.now(),
            size_bytes=1000
        )
        
        small_entry = CacheEntry(
            key="small",
            value="value",
            created_at=datetime.now(),
            size_bytes=100
        )
        
        large_score = CacheStrategy.size_score(large_entry)
        small_score = CacheStrategy.size_score(small_entry)
        
        # Size strategy uses negative bytes, so larger should have lower score
        assert large_score < small_score