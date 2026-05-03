from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from llm.cache import cached_call

# Baseten *Model API* (multi-model inference endpoint, not a per-deployment
# URL). The default model below is the active research-pipeline pin; the
# previous Qwen/Qwen3-235B-A22B dedicated deployment was retired (mid-Step 4
# rebuild) and the project moved to deepseek-ai/DeepSeek-V4-Pro on the same
# Baseten Model API. Both LLM_BASE_URL and LLM_MODEL are env-overridable.
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://inference.baseten.co/v1")
MODEL = os.environ.get("LLM_MODEL", "deepseek-ai/DeepSeek-V4-Pro")

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client
    key = os.environ.get("BASETEN_API_KEY")
    if not key:
        msg = "BASETEN_API_KEY is not set (e.g. in .env for `task run`)."
        raise ValueError(msg)
    _client = OpenAI(
        base_url=LLM_BASE_URL,
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
