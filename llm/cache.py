from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any


def cache_key(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def cached_call(
    cache_dir: Path, payload: dict[str, Any], fetch: Callable[[], Any]
) -> Any:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key(payload)
    path = cache_dir / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    result = fetch()
    path.write_text(json.dumps(result, default=str), encoding="utf-8")
    return result
