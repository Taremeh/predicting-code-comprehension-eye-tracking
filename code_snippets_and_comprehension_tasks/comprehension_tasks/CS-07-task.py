lru_cache = LruCache()
lru_cache.set("key-3945", "This is a sample value.")
print(lru_cache.get("key-3945"))
print(str(lru_cache))