"""Tests for experiments.constraint_quality — synthetic graphs only."""

from __future__ import annotations

import unittest

from constraints.constraint_builder import ClaimRecord
from experiments.constraint_quality import compute_quality


def _claim(
    *,
    var_a: str,
    var_b: str,
    cause: str,
    effect: str,
    confidence: float,
    constraint_type: str,
) -> ClaimRecord:
    return ClaimRecord(
        var_a=var_a,
        var_b=var_b,
        cause=cause,
        effect=effect,
        confidence=confidence,
        constraint_type=constraint_type,
        source="test",
    )


class TestComputeQuality(unittest.TestCase):
    def test_happy_path_precision_recall(self) -> None:
        true_edges = {("A", "B"), ("B", "C")}
        priors = [
            _claim(
                var_a="A",
                var_b="B",
                cause="A",
                effect="B",
                confidence=1.0,
                constraint_type="hard_required",
            )
        ]
        r = compute_quality(
            priors, true_edges, confidence_threshold=0.7, n_total_pairs=6
        )
        self.assertEqual(r["n_predicted_forward"], 1)
        self.assertEqual(r["precision_forward"], 1.0)
        self.assertEqual(r["recall_forward"], 0.5)
        self.assertEqual(r["hallucination_strict"], 0.0)
        self.assertEqual(r["hallucination_loose"], 0.0)
        self.assertEqual(r["true_positives"], [["A", "B"]])

    def test_strict_hallucination(self) -> None:
        true_edges = {("A", "B")}
        priors = [
            _claim(
                var_a="B",
                var_b="A",
                cause="B",
                effect="A",
                confidence=1.0,
                constraint_type="hard_required",
            )
        ]
        r = compute_quality(
            priors, true_edges, confidence_threshold=0.7, n_total_pairs=2
        )
        self.assertEqual(r["precision_forward"], 0.0)
        self.assertEqual(r["recall_forward"], 0.0)
        self.assertEqual(r["hallucination_strict"], 1.0)
        self.assertEqual(r["hallucination_loose"], 0.0)
        self.assertEqual(r["false_positives_strict"], [["B", "A"]])

    def test_loose_hallucination(self) -> None:
        true_edges = {("A", "B")}
        priors = [
            _claim(
                var_a="A",
                var_b="C",
                cause="A",
                effect="C",
                confidence=1.0,
                constraint_type="hard_required",
            )
        ]
        r = compute_quality(
            priors, true_edges, confidence_threshold=0.7, n_total_pairs=3
        )
        self.assertEqual(r["hallucination_strict"], 0.0)
        self.assertEqual(r["hallucination_loose"], 1.0)
        self.assertEqual(r["false_positives_loose"], [["A", "C"]])

    def test_hard_forbidden_reverse_not_forward_and_precision_forbidden(self) -> None:
        true_edges = {("B", "A")}
        priors = [
            _claim(
                var_a="A",
                var_b="B",
                cause="A",
                effect="B",
                confidence=0.9,
                constraint_type="hard_forbidden_reverse",
            )
        ]
        r = compute_quality(
            priors, true_edges, confidence_threshold=0.7, n_total_pairs=2
        )
        self.assertEqual(r["n_predicted_forward"], 0)
        self.assertEqual(r["n_predicted_forbidden"], 1)
        self.assertEqual(r["precision_forbidden"], 0.0)

    def test_threshold_inclusive_at_tau(self) -> None:
        true_edges = {("A", "B")}
        priors = [
            _claim(
                var_a="A",
                var_b="B",
                cause="A",
                effect="B",
                confidence=0.7,
                constraint_type="hard_required",
            )
        ]
        r = compute_quality(
            priors, true_edges, confidence_threshold=0.7, n_total_pairs=2
        )
        self.assertEqual(r["n_predicted_forward"], 1)
        self.assertEqual(r["precision_forward"], 1.0)

    def test_coverage_llm_style(self) -> None:
        priors = [
            _claim(
                var_a="A",
                var_b="B",
                cause="A",
                effect="B",
                confidence=1.0,
                constraint_type="hard_required",
            ),
            _claim(
                var_a="A",
                var_b="C",
                cause="A",
                effect="C",
                confidence=1.0,
                constraint_type="soft_prior",
            ),
            _claim(
                var_a="B",
                var_b="C",
                cause="B",
                effect="C",
                confidence=1.0,
                constraint_type="hard_required",
            ),
        ]
        r = compute_quality(priors, set(), confidence_threshold=0.7, n_total_pairs=6)
        self.assertAlmostEqual(r["coverage"], 0.5)

    def test_empty_forward_predictions(self) -> None:
        true_edges = {("A", "B")}
        r = compute_quality([], true_edges, confidence_threshold=0.7, n_total_pairs=2)
        self.assertEqual(r["n_predicted_forward"], 0)
        self.assertEqual(r["precision_forward"], 0.0)
        self.assertEqual(r["recall_forward"], 0.0)
        self.assertEqual(r["hallucination_strict"], 0.0)
        self.assertEqual(r["hallucination_loose"], 0.0)

    def test_deterministic(self) -> None:
        true_edges = {("A", "B")}
        priors = [
            _claim(
                var_a="B",
                var_b="A",
                cause="B",
                effect="A",
                confidence=0.95,
                constraint_type="soft_prior",
            ),
            _claim(
                var_a="A",
                var_b="B",
                cause="A",
                effect="B",
                confidence=1.0,
                constraint_type="hard_required",
            ),
        ]
        a = compute_quality(
            priors, true_edges, confidence_threshold=0.7, n_total_pairs=2
        )
        b = compute_quality(
            priors, true_edges, confidence_threshold=0.7, n_total_pairs=2
        )
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
