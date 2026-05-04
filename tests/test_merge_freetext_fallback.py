"""Tests for :mod:`reasoning.merge_freetext_fallback`."""

from __future__ import annotations

import unittest

from reasoning.merge_freetext_fallback import merge_with_freetext_fallback


class MergeFreetextFallbackTests(unittest.TestCase):
    def test_substitutes_only_no_context_in_pairs(self) -> None:
        reactome = {
            "pairs": [
                {
                    "var_a": "A",
                    "var_b": "B",
                    "cause": "unknown",
                    "effect": "unknown",
                    "confidence": 0.0,
                    "constraint_type": "no_context",
                },
                {
                    "var_a": "A",
                    "var_b": "C",
                    "cause": "A",
                    "effect": "C",
                    "confidence": 0.95,
                    "constraint_type": "hard_required",
                },
            ],
            "no_context_pairs": [],
        }
        freetext = {
            "pairs": [
                {
                    "var_a": "A",
                    "var_b": "B",
                    "cause": "A",
                    "effect": "B",
                    "confidence": 0.99,
                    "constraint_type": "hard_required",
                    "source": "freetext_llm",
                }
            ]
        }
        col = {"A", "B", "C"}
        out = merge_with_freetext_fallback(reactome, freetext, column_set=col)
        self.assertEqual(out["n_fallback_applied"], 1)
        self.assertEqual(
            out["fallback_applied_pairs"],
            [["A", "B"]],
        )
        sub = next(p for p in out["pairs"] if p["var_a"] == "A" and p["var_b"] == "B")
        self.assertEqual(sub["source"], "freetext_fallback")
        self.assertEqual(sub["cause"], "A")
        self.assertEqual(sub["effect"], "B")
        self.assertAlmostEqual(sub["confidence"], 0.99)
        intact = next(
            p for p in out["pairs"] if p["var_a"] == "A" and p["var_b"] == "C"
        )
        self.assertEqual(intact["constraint_type"], "hard_required")

    def test_unknown_not_overridden(self) -> None:
        reactome = {
            "pairs": [
                {
                    "var_a": "A",
                    "var_b": "B",
                    "cause": "unknown",
                    "effect": "unknown",
                    "confidence": 0.4,
                    "constraint_type": "unknown",
                }
            ],
            "no_context_pairs": [],
        }
        freetext = {
            "pairs": [
                {
                    "var_a": "A",
                    "var_b": "B",
                    "cause": "A",
                    "effect": "B",
                    "confidence": 0.99,
                    "constraint_type": "hard_required",
                    "source": "freetext_llm",
                }
            ]
        }
        out = merge_with_freetext_fallback(reactome, freetext, column_set={"A", "B"})
        self.assertEqual(out["n_fallback_applied"], 0)
        self.assertEqual(out["pairs"][0]["constraint_type"], "unknown")

    def test_missing_freetext_pass_through_no_context(self) -> None:
        reactome = {
            "pairs": [],
            "no_context_pairs": [["X", "Y"]],
        }
        freetext = {"pairs": []}
        out = merge_with_freetext_fallback(reactome, freetext, column_set={"X", "Y"})
        self.assertEqual(out["n_fallback_applied"], 0)
        self.assertEqual(out["no_context_pairs"], [["X", "Y"]])
        self.assertEqual(out["pairs"], [])

    def test_no_context_pairs_promoted_and_counts(self) -> None:
        reactome = {
            "pairs": [],
            "no_context_pairs": [["A", "B"], ["X", "Y"]],
            "n_pairs_total": 6,
            "n_processed": 6,
        }
        freetext = {
            "pairs": [
                {
                    "var_a": "B",
                    "var_b": "A",
                    "cause": "A",
                    "effect": "B",
                    "confidence": 0.8,
                    "constraint_type": "soft_prior",
                    "source": "freetext_llm",
                }
            ]
        }
        out = merge_with_freetext_fallback(
            reactome, freetext, column_set={"A", "B", "X", "Y"}
        )
        self.assertEqual(out["n_no_context"], 1)
        self.assertEqual(len(out["pairs"]), 1)
        self.assertEqual(out["n_with_context"], 1)
        self.assertEqual(out["no_context_pairs"], [["X", "Y"]])
        self.assertEqual(out["n_fallback_applied"], 1)

    def test_recomputes_n_high_confidence(self) -> None:
        reactome = {
            "pairs": [],
            "no_context_pairs": [["A", "B"]],
        }
        freetext = {
            "pairs": [
                {
                    "var_a": "A",
                    "var_b": "B",
                    "cause": "A",
                    "effect": "B",
                    "confidence": 1.0,
                    "constraint_type": "hard_required",
                    "source": "freetext_llm",
                }
            ]
        }
        out = merge_with_freetext_fallback(reactome, freetext, column_set={"A", "B"})
        self.assertEqual(out["n_high_confidence"], 1)


class TwoStageFallbackPipelineTests(unittest.TestCase):
    """Match ``experiments.run_reasoning_with_fallback`` cumulative accounting."""

    def test_paragraph_then_per_pair_fills_disjoint_no_context(self) -> None:
        reactome = {
            "pairs": [
                {
                    "var_a": "A",
                    "var_b": "B",
                    "cause": "unknown",
                    "effect": "unknown",
                    "confidence": 0.0,
                    "constraint_type": "no_context",
                },
                {
                    "var_a": "A",
                    "var_b": "C",
                    "cause": "A",
                    "effect": "C",
                    "confidence": 0.95,
                    "constraint_type": "hard_required",
                    "source": "reactome_llm",
                },
            ],
            "no_context_pairs": [["X", "Y"]],
        }
        paragraph = {
            "pairs": [
                {
                    "var_a": "A",
                    "var_b": "B",
                    "cause": "A",
                    "effect": "B",
                    "confidence": 0.95,
                    "constraint_type": "hard_required",
                    "source": "freetext_llm",
                }
            ]
        }
        per_pair = {
            "pairs": [
                {
                    "var_a": "X",
                    "var_b": "Y",
                    "cause": "X",
                    "effect": "Y",
                    "confidence": 0.95,
                    "constraint_type": "hard_required",
                    "source": "per_pair_freetext_fallback",
                }
            ]
        }
        col = {"A", "B", "C", "X", "Y"}

        mid = merge_with_freetext_fallback(reactome, paragraph, column_set=col)
        n_para = int(mid["n_fallback_applied"])
        fb_total = list(mid["fallback_applied_pairs"] or [])

        final = merge_with_freetext_fallback(mid, per_pair, column_set=col)
        n_pp = int(final["n_fallback_applied"])
        fb_total.extend(list(final["fallback_applied_pairs"] or []))

        final["n_fallback_applied"] = n_para + n_pp
        final["fallback_applied_pairs"] = sorted(
            fb_total,
            key=lambda p: (p[0], p[1]),
        )

        self.assertEqual(n_para, 1)
        self.assertEqual(n_pp, 1)
        self.assertEqual(final["n_fallback_applied"], 2)
        self.assertEqual(
            final["fallback_applied_pairs"],
            [["A", "B"], ["X", "Y"]],
        )
        intact = next(
            p for p in final["pairs"] if p["var_a"] == "A" and p["var_b"] == "C"
        )
        self.assertEqual(intact["constraint_type"], "hard_required")
        self.assertEqual(intact.get("source"), "reactome_llm")
        sub_ab = next(
            p for p in final["pairs"] if p["var_a"] == "A" and p["var_b"] == "B"
        )
        self.assertEqual(sub_ab["source"], "freetext_fallback")
        sub_xy = next(
            p for p in final["pairs"] if p["var_a"] == "X" and p["var_b"] == "Y"
        )
        self.assertEqual(sub_xy["source"], "per_pair_freetext_fallback")


if __name__ == "__main__":
    unittest.main()
