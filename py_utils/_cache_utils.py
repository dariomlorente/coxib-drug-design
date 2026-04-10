from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


def _get_cache_key(*args: Any) -> str:
    """Generate a cache key from arguments using SHA256."""
    key_str = "|".join(str(arg) for arg in args)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def _load_cache(cache_file: Path) -> dict[str, Any]:
    """Load cache from gzip-compressed JSON file."""
    if not cache_file.exists():
        return {}

    try:
        with gzip.open(cache_file, "rt", encoding="utf-8") as f:
            return json.load(f)
    except (gzip.BadGzipFile, json.JSONDecodeError, IOError):
        print(f"⚠️  Cache file corrupted or missing, starting fresh: {cache_file}")
        return {}


def _save_cache(cache_file: Path, cache: dict[str, Any]) -> None:
    """Save cache to gzip-compressed JSON file, safely."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = cache_file.with_suffix(".tmp")
    
    try:
        with gzip.open(temp_file, "wt", encoding="utf-8", compresslevel=6) as f:
            json.dump(cache, f, separators=(",", ":"))
        temp_file.replace(cache_file)
    except Exception as e:
        print(f"⚠️  Warning: Failed to save cache {cache_file.name}: {e}")
        if temp_file.exists():
            temp_file.unlink()
