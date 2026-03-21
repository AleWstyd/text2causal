import matplotlib
import networkx as nx
import unittest

from app_pipeline import (
    calculate_graph_metrics,
    extract_llm_constraints,
    run_algorithm_simulation,
)
from utils.graph_utils import build_graph_figure, build_hierarchical_layout


matplotlib.use("Agg")


class AppPipelineTests(unittest.TestCase):
    def test_extract_llm_constraints_filters_by_threshold(self):
        relations = [
            {"cause": "Smoking", "effect": "Lung_Cancer", "confidence": 0.92},
            {"cause": "Allergy", "effect": "Coughing", "confidence": 0.45},
        ]

        result = extract_llm_constraints(
            variable_names=["Smoking", "Lung_Cancer", "Allergy", "Coughing"],
            background_text="unused",
            threshold=0.7,
            extractor=lambda _variables, _text: relations,
        )

        self.assertEqual(result["relations"], relations)
        self.assertEqual(result["required_edges"], [("Smoking", "Lung_Cancer")])

    def test_calculate_graph_metrics_returns_expected_scores(self):
        graph = nx.DiGraph()
        graph.add_edge("Smoking", "Lung_Cancer")
        graph.add_edge("Genetics", "Attention_Disorder")

        metrics = calculate_graph_metrics(
            graph,
            true_edges=[
                ("Smoking", "Lung_Cancer"),
                ("Genetics", "Attention_Disorder"),
                ("Allergy", "Coughing"),
            ],
        )

        self.assertEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 2 / 3)
        self.assertAlmostEqual(metrics["f1"], 0.8)

    def test_run_algorithm_simulation_returns_both_graphs_and_metrics(self):
        def empty_runner(_data_matrix, variable_names):
            graph = nx.DiGraph()
            graph.add_nodes_from(variable_names)
            return graph

        result = run_algorithm_simulation(
            algorithm_name="PC",
            data_matrix=[[0, 1], [1, 0]],
            variable_names=["Smoking", "Lung_Cancer"],
            required_edges=[("Smoking", "Lung_Cancer")],
            true_edges=[("Smoking", "Lung_Cancer")],
            runners={"PC": empty_runner},
        )

        self.assertEqual(list(result["baseline_graph"].edges()), [])
        self.assertEqual(
            list(result["constrained_graph"].edges()),
            [("Smoking", "Lung_Cancer")],
        )
        self.assertEqual(
            result["baseline_metrics"],
            {
                "precision": 0,
                "recall": 0.0,
                "f1": 0,
            },
        )
        self.assertEqual(
            result["constrained_metrics"],
            {
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
            },
        )

    def test_build_graph_figure_returns_matplotlib_figure(self):
        graph = nx.DiGraph()
        graph.add_edge("Smoking", "Lung_Cancer")

        figure = build_graph_figure(graph, title="Example graph")

        self.assertEqual(figure.axes[0].get_title(), "Example graph")

    def test_build_hierarchical_layout_flows_top_to_bottom(self):
        graph = nx.DiGraph()
        graph.add_edges_from(
            [
                ("Smoking", "Lung_Cancer"),
                ("Lung_Cancer", "Fatigue"),
                ("Peer_Pressure", "Smoking"),
            ]
        )

        positions = build_hierarchical_layout(graph)

        self.assertGreater(positions["Peer_Pressure"][1], positions["Smoking"][1])
        self.assertGreater(positions["Smoking"][1], positions["Lung_Cancer"][1])
        self.assertGreater(positions["Lung_Cancer"][1], positions["Fatigue"][1])


if __name__ == "__main__":
    unittest.main()
