"""Regression tests for the C-LLM-only DAG builder (Step 4 Phase 4)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import networkx as nx

from reasoning.dag_from_priors import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    build_meta,
    predict_dag_from_priors,
    predict_dag_from_priors_with_meta,
)


def _claim(
    cause: str, effect: str, confidence: float, *, ctype: str = "soft_prior"
) -> dict[str, object]:
    return {
        "var_a": cause,
        "var_b": effect,
        "cause": cause,
        "effect": effect,
        "confidence": confidence,
        "constraint_type": ctype,
        "supporting_reactions": [],
        "contradicting_reactions": [],
        "reasoning": "fixture",
    }


class CycleBreakingTests(unittest.TestCase):
    def test_lower_confidence_cycle_closer_is_dropped(self) -> None:
        priors = {
            "pairs": [
                _claim("A", "B", 0.95, ctype="hard_required"),
                _claim("B", "C", 0.90, ctype="hard_required"),
                _claim("C", "A", 0.85, ctype="soft_prior"),
            ]
        }
        result = predict_dag_from_priors_with_meta(
            priors, ["A", "B", "C"], confidence_threshold=0.7
        )
        self.assertTrue(nx.is_directed_acyclic_graph(result.graph))
        self.assertCountEqual(list(result.graph.edges()), [("A", "B"), ("B", "C")])
        dropped = [
            d for d in result.dropped_due_to_cycle if d["reason"] == "would_close_cycle"
        ]
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0]["cause"], "C")
        self.assertEqual(dropped[0]["effect"], "A")
        self.assertEqual(dropped[0]["confidence"], 0.85)


class ThresholdFilterTests(unittest.TestCase):
    def test_below_threshold_claims_dropped(self) -> None:
        priors = {
            "pairs": [
                _claim("A", "B", 0.95),
                _claim("B", "C", 0.69),
                _claim("C", "D", 0.60),
            ]
        }
        result = predict_dag_from_priors_with_meta(
            priors, ["A", "B", "C", "D"], confidence_threshold=0.7
        )
        self.assertCountEqual(list(result.graph.edges()), [("A", "B")])
        self.assertEqual(result.n_after_threshold, 1)

    def test_threshold_default_matches_doc(self) -> None:
        self.assertEqual(DEFAULT_CONFIDENCE_THRESHOLD, 0.7)


class ConstraintTypeFilterTests(unittest.TestCase):
    def test_unknown_constraint_type_dropped_even_above_threshold(self) -> None:
        priors = {
            "pairs": [
                _claim("A", "B", 0.95, ctype="hard_required"),
                _claim("B", "C", 0.95, ctype="unknown"),
                _claim("C", "D", 0.95, ctype="no_context"),
            ]
        }
        result = predict_dag_from_priors_with_meta(
            priors, ["A", "B", "C", "D"], confidence_threshold=0.7
        )
        self.assertCountEqual(list(result.graph.edges()), [("A", "B")])

    def test_hard_forbidden_reverse_kept(self) -> None:
        priors = {
            "pairs": [
                _claim("A", "B", 0.95, ctype="hard_forbidden_reverse"),
            ]
        }
        result = predict_dag_from_priors_with_meta(
            priors, ["A", "B"], confidence_threshold=0.7
        )
        self.assertCountEqual(list(result.graph.edges()), [("A", "B")])


class IsolatedNodePreservationTests(unittest.TestCase):
    def test_all_variable_names_present_even_when_isolated(self) -> None:
        priors = {"pairs": [_claim("A", "B", 0.95, ctype="hard_required")]}
        names = ["A", "B", "C", "D", "E"]
        result = predict_dag_from_priors_with_meta(
            priors, names, confidence_threshold=0.7
        )
        self.assertCountEqual(list(result.graph.nodes()), names)
        self.assertEqual(result.graph.number_of_edges(), 1)

    def test_no_qualifying_priors_yields_node_only_dag(self) -> None:
        priors = {"pairs": []}
        names = ["A", "B", "C"]
        result = predict_dag_from_priors_with_meta(
            priors, names, confidence_threshold=0.7
        )
        self.assertCountEqual(list(result.graph.nodes()), names)
        self.assertEqual(result.graph.number_of_edges(), 0)
        self.assertTrue(nx.is_directed_acyclic_graph(result.graph))


class UnknownDirectionAndOutOfVocabTests(unittest.TestCase):
    def test_unknown_cause_or_effect_skipped(self) -> None:
        priors = {
            "pairs": [
                _claim("unknown", "B", 0.95, ctype="hard_required"),
                _claim("A", "unknown", 0.95, ctype="hard_required"),
                _claim("A", "B", 0.95, ctype="hard_required"),
            ]
        }
        result = predict_dag_from_priors_with_meta(
            priors, ["A", "B"], confidence_threshold=0.7
        )
        self.assertCountEqual(list(result.graph.edges()), [("A", "B")])

    def test_node_not_in_variable_names_dropped_with_reason(self) -> None:
        priors = {
            "pairs": [
                _claim("A", "ZZZ", 0.95, ctype="hard_required"),
                _claim("A", "B", 0.95, ctype="hard_required"),
            ]
        }
        result = predict_dag_from_priors_with_meta(
            priors, ["A", "B"], confidence_threshold=0.7
        )
        self.assertCountEqual(list(result.graph.edges()), [("A", "B")])
        reasons = {d["reason"] for d in result.dropped_due_to_cycle}
        self.assertIn("node_not_in_variable_names", reasons)


class FromFileEntryPointTests(unittest.TestCase):
    def test_predict_dag_from_priors_reads_file_and_returns_graph(self) -> None:
        priors = {"pairs": [_claim("A", "B", 0.95, ctype="hard_required")]}
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "priors.json"
            path.write_text(json.dumps(priors), encoding="utf-8")
            graph = predict_dag_from_priors(path, ["A", "B"])
        self.assertCountEqual(list(graph.edges()), [("A", "B")])


class MetaTests(unittest.TestCase):
    def test_meta_records_threshold_and_hash_and_dropped(self) -> None:
        priors = {
            "pairs": [
                _claim("A", "B", 0.95, ctype="hard_required"),
                _claim("B", "C", 0.90, ctype="hard_required"),
                _claim("C", "A", 0.85, ctype="soft_prior"),
            ]
        }
        result = predict_dag_from_priors_with_meta(
            priors, ["A", "B", "C"], confidence_threshold=0.7
        )
        meta = build_meta(result)
        self.assertEqual(meta["threshold"], 0.7)
        self.assertEqual(meta["n_claims_kept"], 2)
        self.assertEqual(meta["n_dropped_due_to_cycle"], 1)
        self.assertTrue(meta["is_acyclic"])
        self.assertEqual(meta["node_count"], 3)
        self.assertEqual(meta["edge_count"], 2)
        self.assertEqual(len(meta["source_priors_hash"]), 64)


if __name__ == "__main__":
    unittest.main()
