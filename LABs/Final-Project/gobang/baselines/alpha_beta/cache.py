"""
FIFO Cache implementation for transposition table.
Ported from https://github.com/lihongxun945/gobang
"""

from .config import config


class Cache:
    """FIFO cache for storing transposition table entries."""
    
    def __init__(self, capacity=1000000):
        self.capacity = capacity
        self.cache = []
        self.map = {}
    
    def get(self, key):
        """Get a value by key."""
        if not config.enable_cache:
            return None
        return self.map.get(key, None)
    
    def put(self, key, value):
        """Set or insert a value."""
        if not config.enable_cache:
            return False
        
        if len(self.cache) >= self.capacity:
            oldest_key = self.cache.pop(0)  # Remove oldest key
            if oldest_key in self.map:
                del self.map[oldest_key]
        
        if key not in self.map:
            self.cache.append(key)
        self.map[key] = value
        return True
    
    def has(self, key):
        """Check if key exists in cache."""
        if not config.enable_cache:
            return False
        return key in self.map
