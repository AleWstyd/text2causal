from __future__ import annotations

import unittest

import networkx as nx

from evaluation.harness import (
    evaluate,
    recompute_metrics_from_predicted_edges,
)


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
        self.assertEqual(metrics["directed_precision"], 1.0)
        self.assertEqual(metrics["directed_recall"], 1.0)
        self.assertEqual(metrics["directed_f1"], 1.0)
        self.assertEqual(metrics["cpdag_precision"], 1.0)
        self.assertEqual(metrics["cpdag_recall"], 1.0)
        self.assertEqual(metrics["cpdag_f1"], 1.0)
        self.assertEqual(metrics["shd_cpdag"], 0.0)

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
        self.assertEqual(metrics["directed_precision"], 1.0)
        self.assertEqual(metrics["directed_recall"], 0.5)
        self.assertAlmostEqual(metrics["directed_f1"], 2 / 3)
        self.assertAlmostEqual(metrics["cpdag_f1"], 2 / 3)

    def test_evaluate_reversed_edge_skeleton_one_directed_zero(self) -> None:
        true_graph = nx.DiGraph()
        true_graph.add_edge("A", "B")
        predicted_graph = nx.DiGraph()
        predicted_graph.add_edge("B", "A")

        metrics = evaluate(predicted_graph, true_graph)

        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["directed_f1"], 0.0)
        self.assertEqual(metrics["cpdag_f1"], 0.0)

    def test_recompute_metrics_from_row_edges(self) -> None:
        true_graph = nx.DiGraph()
        true_graph.add_edges_from([("A", "B"), ("B", "C")])
        variable_names = ["A", "B", "C"]
        row: dict = {
            "status": "ok",
            "predicted_edges": [["A", "B"]],
            "metrics": {"f1": 0.0},
        }
        recompute_metrics_from_predicted_edges(row, variable_names, true_graph)
        self.assertIn("directed_f1", row["metrics"])
        g = nx.DiGraph()
        g.add_nodes_from(variable_names)
        g.add_edge("A", "B")
        expected = evaluate(g, true_graph)
        for k in (
            "directed_precision",
            "directed_recall",
            "directed_f1",
            "cpdag_precision",
            "cpdag_recall",
            "cpdag_f1",
            "shd_cpdag",
        ):
            self.assertAlmostEqual(row["metrics"][k], expected[k], places=6)

    def test_mutual_arc_encoding_cpdag_credit_vs_directed_f1(self) -> None:
        true_graph = nx.DiGraph()
        true_graph.add_edge("A", "B")
        predicted_graph = nx.DiGraph()
        predicted_graph.add_edge("A", "B")
        predicted_graph.add_edge("B", "A")
        metrics = evaluate(predicted_graph, true_graph)
        self.assertAlmostEqual(metrics["directed_precision"], 0.5)
        self.assertAlmostEqual(metrics["directed_recall"], 1.0)
        self.assertAlmostEqual(metrics["cpdag_precision"], 0.5)
        self.assertAlmostEqual(metrics["cpdag_recall"], 0.5)
        self.assertGreater(metrics["directed_f1"], metrics["cpdag_f1"])

    def test_evaluate_chain_fully_directed_cpdag_equals_directed(self) -> None:
        true_graph = nx.DiGraph()
        true_graph.add_edges_from([("A", "B"), ("B", "C")])
        predicted_graph = true_graph.copy()
        metrics = evaluate(predicted_graph, true_graph)
        self.assertEqual(metrics["cpdag_f1"], metrics["directed_f1"])


if __name__ == "__main__":
    unittest.main()
