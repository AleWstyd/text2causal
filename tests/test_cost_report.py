"""Regression tests for Step 6 Phase 5 LLM cache cost reporting."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from experiments import cost_report


def _response(
    content: str,
    *,
    model: str = "deepseek-ai/DeepSeek-V4-Pro",
    pt: int = 0,
    ct: int = 0,
) -> dict:
    total = pt + ct if pt or ct else 0
    return {
        "choices": [
            {
                "message": {
                    "content": content,
                    "role": "assistant",
                },
            }
        ],
        "model": model,
        "usage": {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": total,
        },
    }


class ClassifyStageTests(unittest.TestCase):
    def test_grounding_single_inner_shape(self) -> None:
        content = json.dumps(
            {
                "kind": "protein",
                "ids": ["P12345"],
                "canonical_name": "x",
                "gene_names": ["MAPK1"],
            }
        )
        self.assertEqual(cost_report.classify_stage(_response(content)), "grounding")

    def test_grounding_batched_columns(self) -> None:
        content = json.dumps(
            {
                "col_a": {
                    "kind": "protein",
                    "ids": ["P0"],
                    "canonical_name": "A",
                    "gene_names": [],
                },
                "col_b": {
                    "kind": "family",
                    "gene_names": ["JNK"],
                    "canonical_name": "B",
                },
            }
        )
        self.assertEqual(cost_report.classify_stage(_response(content)), "grounding")

    def test_reasoning(self) -> None:
        content = json.dumps(
            {
                "cause": "a",
                "effect": "b",
                "confidence": 0.5,
                "constraint_type": "hard_required",
                "supporting_reactions": ["R-HSA-1"],
                "contradicting_reactions": [],
            }
        )
        self.assertEqual(cost_report.classify_stage(_response(content)), "reasoning")

    def test_reasoning_contradicting_only(self) -> None:
        content = json.dumps(
            {
                "cause": "a",
                "effect": "b",
                "confidence": 0.5,
                "constraint_type": "soft_prior",
                "contradicting_reactions": [],
            }
        )
        self.assertEqual(cost_report.classify_stage(_response(content)), "reasoning")

    def test_freetext(self) -> None:
        content = json.dumps(
            [
                {
                    "cause": "x",
                    "effect": "y",
                    "relation_type": "required",
                    "confidence": 1.0,
                }
            ]
        )
        self.assertEqual(cost_report.classify_stage(_response(content)), "freetext")

    def test_unknown_bad_json(self) -> None:
        self.assertEqual(cost_report.classify_stage(_response("not-json")), "unknown")

    def test_unknown_wrong_shape(self) -> None:
        content = json.dumps({"foo": 1})
        self.assertEqual(cost_report.classify_stage(_response(content)), "unknown")

    def test_fenced_json_grounding(self) -> None:
        inner = json.dumps(
            {
                "kind": "metabolite",
                "ids": None,
                "gene_names": [],
                "canonical_name": "PIP2",
            }
        )
        wrapped = f"```json\n{inner}\n```"
        self.assertEqual(cost_report.classify_stage(_response(wrapped)), "grounding")


class WalkCacheTests(unittest.TestCase):
    def test_skips_non_json_files(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "notes.txt").write_text("hello", encoding="utf-8")
            self.assertEqual(cost_report.walk_cache(root), [])

    def test_skips_invalid_json(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "broken.json").write_text("{", encoding="utf-8")
            self.assertEqual(cost_report.walk_cache(root), [])

    def test_collects_valid_cache_file(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = _response(
                '{"cause":"a","effect":"b","confidence":1,"constraint_type":"x","supporting_reactions":[]}',
                pt=3,
                ct=4,
            )
            (root / "a.json").write_text(json.dumps(payload), encoding="utf-8")
            entries = cost_report.walk_cache(root)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["prompt_tokens"], 3)
            self.assertEqual(entries[0]["completion_tokens"], 4)


class AggregateTests(unittest.TestCase):
    def test_sums_tokens_and_cost(self) -> None:
        entries = [
            {
                "stage": "reasoning",
                "served_model": "m1",
                "prompt_tokens": 1_000_000,
                "completion_tokens": 2_000_000,
                "total_tokens": 3_000_000,
                "estimated_cost_usd": cost_report.estimated_cost_usd(
                    1_000_000, 2_000_000
                ),
            },
            {
                "stage": "grounding",
                "served_model": "m1",
                "prompt_tokens": 500_000,
                "completion_tokens": 0,
                "total_tokens": 500_000,
                "estimated_cost_usd": cost_report.estimated_cost_usd(500_000, 0),
            },
        ]
        agg = cost_report.aggregate(entries)
        t = agg["totals"]
        self.assertEqual(t["n_entries"], 2)
        self.assertEqual(t["prompt_tokens"], 1_500_000)
        self.assertEqual(t["completion_tokens"], 2_000_000)
        expect = cost_report.estimated_cost_usd(1_500_000, 2_000_000)
        self.assertAlmostEqual(t["estimated_cost_usd"], expect, places=9)
        self.assertEqual(agg["by_stage"]["reasoning"]["n_entries"], 1)
        self.assertEqual(agg["by_stage"]["grounding"]["n_entries"], 1)

    def test_cost_formula_matches_constants(self) -> None:
        ipt, opt = 2_000_000, 3_000_000
        got = cost_report.estimated_cost_usd(ipt, opt)
        want = (
            2 * cost_report.PRICE_PER_M_INPUT_USD
            + 3 * cost_report.PRICE_PER_M_OUTPUT_USD
        )
        self.assertAlmostEqual(got, want, places=12)


class JsonRoundTripTests(unittest.TestCase):
    def test_output_roundtrips(self) -> None:
        doc = {
            "snapshot_date": cost_report.PRICE_SNAPSHOT_DATE,
            "price_per_m_input_usd": cost_report.PRICE_PER_M_INPUT_USD,
            "price_per_m_output_usd": cost_report.PRICE_PER_M_OUTPUT_USD,
            "price_source_url": cost_report.PRICE_SOURCE_URL,
            "summary": cost_report.aggregate([]),
            "approx_wall_clock_seconds": {
                "grounding": 0,
                "reasoning": 0,
                "freetext": 0,
                "total": 0,
            },
            "latency_disclosure": "x",
            "headline": "y",
        }
        dumped = json.dumps(doc, indent=2, sort_keys=True)
        back = json.loads(dumped)
        self.assertIn("summary", back)


class MainSmokeTests(unittest.TestCase):
    def test_main_writes_artefact(self) -> None:
        grounding = json.dumps(
            {
                "c": {
                    "kind": "protein",
                    "ids": ["P0"],
                    "canonical_name": "z",
                    "gene_names": [],
                }
            }
        )
        reason = json.dumps(
            {
                "cause": "a",
                "effect": "b",
                "confidence": 0.0,
                "constraint_type": "unknown",
                "supporting_reactions": [],
                "contradicting_reactions": [],
            }
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache" / "llm"
            cache.mkdir(parents=True)
            (cache / "z1.json").write_text(
                json.dumps(_response(grounding, pt=10, ct=20)), encoding="utf-8"
            )
            (cache / "z2.json").write_text(
                json.dumps(_response(reason, pt=100, ct=200, model="other-model")),
                encoding="utf-8",
            )
            out = root / "experiments" / "cost_report.json"
            with patch.object(cost_report, "REPO_ROOT", root):
                with patch("builtins.print"):
                    cost_report.main()
            self.assertTrue(out.is_file())
            data = json.loads(out.read_text(encoding="utf-8"))
            for key in (
                "snapshot_date",
                "price_per_m_input_usd",
                "price_per_m_output_usd",
                "price_source_url",
                "summary",
                "approx_wall_clock_seconds",
                "latency_disclosure",
                "headline",
            ):
                self.assertIn(key, data)


if __name__ == "__main__":
    unittest.main()
