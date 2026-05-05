"""Tests for :mod:`evaluation.cpdag_metrics`."""

from __future__ import annotations

import unittest

from evaluation.cpdag_metrics import (
    cpdag_precision_recall_f1,
    shd_cpdag,
    split_cpdag_edges,
)


class CpdagPrecisionRecallTests(unittest.TestCase):
    def test_all_directed_matches_directed_scoring(self) -> None:
        true = [("A", "B"), ("B", "C")]
        pred_d = [("A", "B"), ("B", "C")]
        cp = cpdag_precision_recall_f1(pred_d, [], true)
        self.assertAlmostEqual(cp["cpdag_precision"], 1.0)
        self.assertAlmostEqual(cp["cpdag_recall"], 1.0)
        self.assertAlmostEqual(cp["cpdag_f1"], 1.0)

    def test_all_undirected_half_credit(self) -> None:
        true = [("A", "B")]
        cp = cpdag_precision_recall_f1([], [("A", "B")], true)
        self.assertAlmostEqual(cp["cpdag_precision"], 0.5)
        self.assertAlmostEqual(cp["cpdag_recall"], 0.5)
        self.assertAlmostEqual(cp["cpdag_f1"], 0.5)

    def test_mixed_directed_and_undirected(self) -> None:
        true = [("A", "B"), ("B", "C")]
        cp = cpdag_precision_recall_f1([("A", "B")], [("B", "C")], true)
        self.assertAlmostEqual(cp["cpdag_precision"], 0.75)
        self.assertAlmostEqual(cp["cpdag_recall"], 0.75)

    def test_wrong_directed_zero_credit(self) -> None:
        true = [("A", "B")]
        cp = cpdag_precision_recall_f1([("B", "A")], [], true)
        self.assertAlmostEqual(cp["cpdag_precision"], 0.0)
        self.assertAlmostEqual(cp["cpdag_recall"], 0.0)

    def test_partially_correct_cpdag_subset(self) -> None:
        true = [("A", "B"), ("B", "C"), ("A", "C")]
        pred = [("A", "B"), ("B", "C")]
        cp = cpdag_precision_recall_f1(pred, [], true)
        self.assertAlmostEqual(cp["cpdag_recall"], 2 / 3)
        self.assertAlmostEqual(cp["cpdag_precision"], 1.0)

    def test_empty_pred(self) -> None:
        cp = cpdag_precision_recall_f1([], [], [("A", "B")])
        self.assertAlmostEqual(cp["cpdag_precision"], 0.0)
        self.assertAlmostEqual(cp["cpdag_recall"], 0.0)

    def test_empty_true(self) -> None:
        cp = cpdag_precision_recall_f1([("A", "B")], [], [])
        self.assertAlmostEqual(cp["cpdag_precision"], 0.0)
        self.assertAlmostEqual(cp["cpdag_recall"], 0.0)

    def test_both_empty(self) -> None:
        cp = cpdag_precision_recall_f1([], [], [])
        self.assertAlmostEqual(cp["cpdag_f1"], 0.0)

    def test_mutual_arcs_normalize_to_undirected(self) -> None:
        true = [("X", "Y")]
        cp = cpdag_precision_recall_f1([("X", "Y"), ("Y", "X")], [], true)
        self.assertAlmostEqual(cp["cpdag_precision"], 0.5)
        self.assertAlmostEqual(cp["cpdag_recall"], 0.5)

    def test_isolated_nodes_no_edges(self) -> None:
        self.assertEqual(shd_cpdag([], [], []), 0)


class SplitCpdagEdgesTests(unittest.TestCase):
    def test_split_directed_and_mutual(self) -> None:
        d, u = split_cpdag_edges([("A", "B"), ("B", "A"), ("A", "C")])
        self.assertEqual(d, [("A", "C")])
        self.assertEqual(u, [("A", "B")])


class ShdCpdagTests(unittest.TestCase):
    def test_perfect_dag_agreement(self) -> None:
        self.assertEqual(
            shd_cpdag([("A", "B")], [], [("A", "B")]),
            0,
        )

    def test_undirected_vs_directed_mismatch_one_not_two(self) -> None:
        # Gold A->B; pred undirected — one SHD unit, not naive matrix 2
        d = shd_cpdag([], [("A", "B")], [("A", "B")])
        self.assertEqual(d, 1)

    def test_reversed_directed(self) -> None:
        d = shd_cpdag([("B", "A")], [], [("A", "B")])
        self.assertEqual(d, 1)

    def test_extra_undirected_edge(self) -> None:
        d = shd_cpdag([], [("A", "B")], [])
        self.assertEqual(d, 1)


if __name__ == "__main__":
    unittest.main()
