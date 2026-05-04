"""Tests for :mod:`reasoning.freetext_fallback`."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from reasoning.freetext_fallback import generate_per_pair_freetext_priors


def _fake_response(
    body: dict[str, object], *, model: str = "deepseek-ai/DeepSeek-V4-Pro"
) -> dict[str, object]:
    return {
        "model": model,
        "choices": [{"message": {"content": json.dumps(body)}}],
    }


class PerPairFreetextFallbackTests(unittest.TestCase):
    def test_dedupes_unordered_pairs_single_llm_call(self) -> None:
        calls: list[list[dict[str, str]]] = []

        def fetch(messages: list[dict[str, str]], cache_dir: Path) -> dict[str, object]:
            calls.append(messages)
            return _fake_response(
                {
                    "cause": "a",
                    "effect": "b",
                    "confidence": 0.95,
                    "constraint_type": "hard_required",
                    "reasoning": "Test.",
                }
            )

        out = generate_per_pair_freetext_priors(
            "t",
            "desc",
            [("b", "a"), ("a", "b")],
            ["a", "b"],
            llm_fetch=fetch,
            cache_dir=Path("/tmp"),
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(out["n_pairs_queried"], 1)
        self.assertEqual(out["n_relations_emitted"], 1)
        self.assertEqual(out["pairs"][0]["var_a"], "a")
        self.assertEqual(out["pairs"][0]["var_b"], "b")
        self.assertEqual(out["pairs"][0]["source"], "per_pair_freetext_fallback")
        self.assertEqual(out["source_kind"], "per_pair_freetext_fallback")

    def test_skips_out_of_vocab_tokens(self) -> None:
        def fetch(messages: list[dict[str, str]], cache_dir: Path) -> dict[str, object]:
            return _fake_response(
                {
                    "cause": "not_a_token",
                    "effect": "a",
                    "confidence": 0.99,
                    "constraint_type": "hard_required",
                    "reasoning": "Bad.",
                }
            )

        out = generate_per_pair_freetext_priors(
            "t",
            "d",
            [("a", "b")],
            ["a", "b"],
            llm_fetch=fetch,
            cache_dir=Path("/tmp"),
        )
        self.assertEqual(out["n_relations_emitted"], 0)
        self.assertEqual(out["n_relations_skipped_unknown_var"], 1)
        self.assertEqual(out["pairs"], [])

    def test_skips_both_unknown_cause_effect(self) -> None:
        def fetch(messages: list[dict[str, str]], cache_dir: Path) -> dict[str, object]:
            return _fake_response(
                {
                    "cause": "unknown",
                    "effect": "unknown",
                    "confidence": 0.1,
                    "constraint_type": "unknown",
                    "reasoning": "x",
                }
            )

        out = generate_per_pair_freetext_priors(
            "t",
            "d",
            [("a", "b")],
            ["a", "b"],
            llm_fetch=fetch,
            cache_dir=Path("/tmp"),
        )
        self.assertEqual(out["n_relations_emitted"], 0)
        self.assertEqual(out["n_relations_skipped_unknown_var"], 1)

    def test_skips_when_only_one_endpoint_unknown(self) -> None:
        def fetch(messages: list[dict[str, str]], cache_dir: Path) -> dict[str, object]:
            return _fake_response(
                {
                    "cause": "unknown",
                    "effect": "b",
                    "confidence": 0.95,
                    "constraint_type": "hard_required",
                    "reasoning": "Incomplete.",
                }
            )

        out = generate_per_pair_freetext_priors(
            "t",
            "d",
            [("a", "b")],
            ["a", "b"],
            llm_fetch=fetch,
            cache_dir=Path("/tmp"),
        )
        self.assertEqual(out["n_relations_emitted"], 0)
        self.assertEqual(out["n_relations_skipped_unknown_var"], 1)

    def test_low_confidence_downgrades_to_unknown(self) -> None:
        def fetch(messages: list[dict[str, str]], cache_dir: Path) -> dict[str, object]:
            return _fake_response(
                {
                    "cause": "a",
                    "effect": "b",
                    "confidence": 0.35,
                    "constraint_type": "hard_required",
                    "reasoning": "Weak.",
                }
            )

        out = generate_per_pair_freetext_priors(
            "t",
            "d",
            [("a", "b")],
            ["a", "b"],
            llm_fetch=fetch,
            cache_dir=Path("/tmp"),
        )
        self.assertEqual(out["n_relations_emitted"], 1)
        self.assertEqual(out["pairs"][0]["constraint_type"], "unknown")
        self.assertEqual(out["n_unknown"], 1)
        self.assertEqual(out["n_hard_required"], 0)

    def test_served_models_collected(self) -> None:
        def fetch(messages: list[dict[str, str]], cache_dir: Path) -> dict[str, object]:
            return _fake_response(
                {
                    "cause": "a",
                    "effect": "b",
                    "confidence": 0.95,
                    "constraint_type": "hard_required",
                    "reasoning": "ok",
                },
                model="deepseek-ai/DeepSeek-V4-Pro",
            )

        out = generate_per_pair_freetext_priors(
            "t",
            "d",
            [("a", "b")],
            ["a", "b"],
            llm_fetch=fetch,
            cache_dir=Path("/tmp"),
        )
        self.assertEqual(out["served_models"], ["deepseek-ai/DeepSeek-V4-Pro"])
        self.assertEqual(
            out["pairs"][0]["served_models"],
            ["deepseek-ai/DeepSeek-V4-Pro"],
        )


if __name__ == "__main__":
    unittest.main()
