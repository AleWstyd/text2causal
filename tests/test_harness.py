from __future__ import annotations

import unittest

import networkx as nx

from evaluation.harness import evaluate


class HarnessTests(unittest.TestCase):
    def test_evaluate_returns_perfect_scores_for_identical_graphs(self) -> None:
        true_graph = nx.DiGraph()
        true_graph.add_edges_from([("A", "B"), ("B", "C")])
        predicted_graph = true_graph.copy()

        metrics = evaluate(predicted_graph, true_graph)

        self.assertEqual(metrics["shd"], 0.0)
        self.assertAlmostEqual(metrics["aupr"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)

    def test_evaluate_scores_missing_edge_and_aligns_nodes(self) -> None:
        true_graph = nx.DiGraph()
        true_graph.add_edges_from([("A", "B"), ("B", "C")])
        predicted_graph = nx.DiGraph()
        predicted_graph.add_edge("A", "B")

        metrics = evaluate(predicted_graph, true_graph)

        self.assertEqual(metrics["shd"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
