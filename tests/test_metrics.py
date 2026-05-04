"""Tests for ``evaluation.metrics`` skeleton vs directed edge overlap."""

from __future__ import annotations

import unittest

from evaluation.metrics import (
    directed_f1,
    directed_precision,
    directed_recall,
    f1_score,
    precision,
    recall,
)


class DirectedVsSkeletonTests(unittest.TestCase):
    def test_reverse_edge_counts_for_skeleton_but_not_directed(self) -> None:
        true = [("A", "B")]
        predicted = [("B", "A")]
        self.assertEqual(precision(predicted, true), 1.0)
        self.assertEqual(recall(predicted, true), 1.0)
        self.assertEqual(
            f1_score(precision(predicted, true), recall(predicted, true)), 1.0
        )
        self.assertEqual(directed_precision(predicted, true), 0.0)
        self.assertEqual(directed_recall(predicted, true), 0.0)
        self.assertEqual(directed_f1(predicted, true), 0.0)

    def test_mixed_orientation_skeleton_one_directed_half(self) -> None:
        true = [("A", "B"), ("C", "D")]
        predicted = [("B", "A"), ("C", "D")]
        self.assertEqual(
            f1_score(precision(predicted, true), recall(predicted, true)), 1.0
        )
        self.assertAlmostEqual(directed_f1(predicted, true), 0.5)

    def test_identical_directed_graphs_match_skeleton_and_directed(self) -> None:
        edges = [("A", "B"), ("C", "D")]
        p = precision(edges, edges)
        r = recall(edges, edges)
        self.assertEqual(p, 1.0)
        self.assertEqual(r, 1.0)
        self.assertEqual(directed_precision(edges, edges), 1.0)
        self.assertEqual(directed_recall(edges, edges), 1.0)
        self.assertEqual(directed_f1(edges, edges), 1.0)


if __name__ == "__main__":
    unittest.main()
