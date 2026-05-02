from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from llm.cache import cached_call

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "openrouter/free"

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client
    key = os.environ.get("OPEN_ROUTER_API_KEY")
    if not key:
        msg = "OPEN_ROUTER_API_KEY is not set (e.g. in .env for `task run`)."
        raise ValueError(msg)
    _client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=key,
    )
    return _client


def chat_completion(
    messages: list[dict[str, Any]],
    *,
    model: str = MODEL,
    cache_dir: Path = Path("cache/llm"),
    **kwargs: Any,
) -> dict[str, Any]:
    if kwargs.get("stream"):
        raise ValueError("Streaming responses cannot be cached by chat_completion().")

    payload = {"model": model, "messages": messages, **kwargs}

    def fetch() -> dict[str, Any]:
        response = _get_client().chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )
        return response.model_dump()

    return cached_call(cache_dir, payload, fetch)
