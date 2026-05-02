from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from llm.cache import cache_key, cached_call


class CacheTests(unittest.TestCase):
    def test_cache_miss_invokes_fetch_and_writes_file(self) -> None:
        calls = 0

        def fetch() -> dict[str, object]:
            nonlocal calls
            calls += 1
            return {"value": 42}

        with TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            payload = {
                "model": "openrouter/free",
                "messages": [{"role": "user", "content": "hi"}],
            }

            result = cached_call(cache_dir, payload, fetch)

            self.assertEqual(result, {"value": 42})
            self.assertEqual(calls, 1)
            self.assertEqual(len(list(cache_dir.glob("*.json"))), 1)

    def test_cache_hit_replays_without_invoking_fetch(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            payload = {
                "model": "openrouter/free",
                "messages": [{"role": "user", "content": "hi"}],
            }
            first = cached_call(cache_dir, payload, lambda: {"value": 42})

            def unexpected_fetch() -> dict[str, object]:
                raise AssertionError("fetch should not be called on cache hit")

            second = cached_call(cache_dir, payload, unexpected_fetch)

            self.assertEqual(second, first)

    def test_cache_key_is_deterministic_for_dict_order(self) -> None:
        payload_a = {
            "model": "openrouter/free",
            "temperature": 0,
            "messages": [{"role": "user", "content": "hi"}],
        }
        payload_b = {
            "messages": [{"content": "hi", "role": "user"}],
            "temperature": 0,
            "model": "openrouter/free",
        }

        self.assertEqual(cache_key(payload_a), cache_key(payload_b))

    def test_cache_key_changes_with_generation_kwargs(self) -> None:
        payload_a = {"model": "openrouter/free", "messages": [], "temperature": 0}
        payload_b = {"model": "openrouter/free", "messages": [], "temperature": 0.7}

        self.assertNotEqual(cache_key(payload_a), cache_key(payload_b))

    def test_cached_response_round_trips_served_model_and_usage(self) -> None:
        response = {
            "id": "chatcmpl-test",
            "model": "qwen/qwen-2.5-72b-instruct",
            "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
            "choices": [{"message": {"content": "[]"}}],
        }

        with TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            payload = {"model": "openrouter/free", "messages": []}
            cached_call(cache_dir, payload, lambda: response)
            replayed = cached_call(cache_dir, payload, lambda: {"model": "wrong"})

            self.assertEqual(replayed["model"], "qwen/qwen-2.5-72b-instruct")
            self.assertEqual(replayed["usage"]["total_tokens"], 16)


if __name__ == "__main__":
    unittest.main()
