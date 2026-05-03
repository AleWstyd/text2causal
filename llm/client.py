from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from llm.cache import cached_call

# Baseten dedicated deployment hosting Qwen3-235B-A22B. The base URL is
# deployment-specific (the `model-XXXXXXXX` segment is the deployment id);
# override with the LLM_BASE_URL env var if the deployment is recreated.
LLM_BASE_URL = os.environ.get(
    "LLM_BASE_URL",
    "https://model-qrjvn993.api.baseten.co/environments/production/sync/v1",
)
MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen3-235B-A22B")

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
