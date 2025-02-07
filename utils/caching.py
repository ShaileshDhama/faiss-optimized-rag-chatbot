import pickle
import os
from config import CACHE_FILE

# Load or initialize cache
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return {}

# Save cache
def save_cache(cache):
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)

# Retrieve from cache
def get_cached_response(query):
    cache = load_cache()
    return cache.get(query)

# Store in cache
def cache_response(query, response):
    cache = load_cache()
    cache[query] = response
    save_cache(cache)
