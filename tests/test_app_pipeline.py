import matplotlib
import networkx as nx
import unittest

import app_pipeline
from app_pipeline import (
    calculate_graph_metrics,
    extract_llm_constraints,
    run_algorithm_simulation,
)
from main import build_relations_frame
from constraints.constraint_builder import (
    PriorKnowledge,
    build_lingam_prior_knowledge,
    build_pc_background_knowledge,
    validate_prior_knowledge,
)
from causallearn.graph.GraphNode import GraphNode
from utils.graph_utils import build_graph_figure, build_hierarchical_layout


matplotlib.use("Agg")


class AppPipelineTests(unittest.TestCase):
    def test_extract_llm_constraints_filters_by_threshold(self):
        relations = [
            {
                "cause": "Smoking",
                "effect": "Lung_Cancer",
                "confidence": 0.92,
                "relation_type": "required",
            },
            {
                "cause": "Car_Accident",
                "effect": "Smoking",
                "confidence": 0.87,
                "relation_type": "forbidden",
            },
            {
                "cause": "Allergy",
                "effect": "Coughing",
                "confidence": 0.45,
                "relation_type": "required",
            },
        ]

        result = extract_llm_constraints(
            variable_names=[
                "Smoking",
                "Lung_Cancer",
                "Allergy",
                "Coughing",
                "Car_Accident",
            ],
            background_text="unused",
            threshold=0.7,
            extractor=lambda _variables, _text: relations,
        )

        self.assertEqual(result["relations"], relations)
        self.assertEqual(
            result["prior_knowledge"],
            PriorKnowledge(
                required_edges=[("Smoking", "Lung_Cancer")],
                forbidden_edges=[("Car_Accident", "Smoking")],
            ),
        )

    def test_extract_llm_constraints_defaults_missing_relation_type_to_required(self):
        relations = [
            {"cause": "Smoking", "effect": "Lung_Cancer", "confidence": 0.92},
        ]

        result = extract_llm_constraints(
            variable_names=["Smoking", "Lung_Cancer"],
            background_text="unused",
            threshold=0.7,
            extractor=lambda _variables, _text: relations,
        )

        self.assertEqual(
            result["prior_knowledge"],
            PriorKnowledge(
                required_edges=[("Smoking", "Lung_Cancer")],
                forbidden_edges=[],
            ),
        )

    def test_extract_llm_constraints_rejects_invalid_relation_type(self):
        relations = [
            {
                "cause": "Smoking",
                "effect": "Lung_Cancer",
                "confidence": 0.92,
                "relation_type": "unsupported",
            },
        ]

        with self.assertRaisesRegex(ValueError, "Relation type must be"):
            extract_llm_constraints(
                variable_names=["Smoking", "Lung_Cancer"],
                background_text="unused",
                threshold=0.7,
                extractor=lambda _variables, _text: relations,
            )

    def test_validate_prior_knowledge_rejects_conflicting_required_and_forbidden_edges(
        self,
    ):
        prior_knowledge = PriorKnowledge(
            required_edges=[("Smoking", "Lung_Cancer")],
            forbidden_edges=[("Smoking", "Lung_Cancer")],
        )

        with self.assertRaisesRegex(ValueError, "both required and forbidden"):
            validate_prior_knowledge(prior_knowledge, ["Smoking", "Lung_Cancer"])

    def test_build_pc_background_knowledge_marks_required_and_forbidden_edges(self):
        prior_knowledge = PriorKnowledge(
            required_edges=[("Smoking", "Lung_Cancer")],
            forbidden_edges=[("Allergy", "Coughing")],
        )

        background_knowledge = build_pc_background_knowledge(prior_knowledge)

        self.assertTrue(
            background_knowledge.is_required(
                GraphNode("Smoking"),
                GraphNode("Lung_Cancer"),
            )
        )
        self.assertTrue(
            background_knowledge.is_forbidden(
                GraphNode("Allergy"),
                GraphNode("Coughing"),
            )
        )

    def test_build_lingam_prior_knowledge_encodes_required_and_forbidden_edges(self):
        prior_knowledge = PriorKnowledge(
            required_edges=[("Smoking", "Lung_Cancer")],
            forbidden_edges=[("Allergy", "Coughing")],
        )

        lingam_prior = build_lingam_prior_knowledge(
            prior_knowledge,
            ["Smoking", "Lung_Cancer", "Allergy", "Coughing"],
        )

        self.assertEqual(lingam_prior[0, 1], 1)
        self.assertEqual(lingam_prior[2, 3], 0)
        self.assertEqual(lingam_prior[1, 0], -1)

    def test_validate_prior_knowledge_rejects_unknown_variables(self):
        prior_knowledge = PriorKnowledge(
            required_edges=[("Smoking", "Unknown_Node")],
            forbidden_edges=[],
        )

        with self.assertRaisesRegex(ValueError, "unknown variables"):
            validate_prior_knowledge(prior_knowledge, ["Smoking", "Lung_Cancer"])

    def test_calculate_graph_metrics_returns_expected_scores(self):
        graph = nx.DiGraph()
        graph.add_nodes_from(
            ["Smoking", "Lung_Cancer", "Genetics", "Attention_Disorder"]
        )
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
        self.assertAlmostEqual(metrics["aupr"], 0.8472222222222222)
        self.assertEqual(metrics["shd"], 1.0)

    def test_calculate_graph_metrics_uses_aupr_and_shd_helpers(self):
        graph = nx.DiGraph()
        graph.add_nodes_from(["Smoking", "Lung_Cancer"])
        graph.add_edge("Smoking", "Lung_Cancer")
        observed_calls = []

        original_aupr = app_pipeline.precision_recall_aupr
        original_shd = app_pipeline.shd

        def fake_aupr(target_graph, predicted_graph):
            observed_calls.append(
                ("aupr", sorted(target_graph.nodes()), sorted(predicted_graph.nodes()))
            )
            return 0.42

        def fake_shd(target_graph, predicted_graph):
            observed_calls.append(
                ("shd", sorted(target_graph.nodes()), sorted(predicted_graph.nodes()))
            )
            return 3.0

        app_pipeline.precision_recall_aupr = fake_aupr
        app_pipeline.shd = fake_shd
        try:
            metrics = calculate_graph_metrics(
                graph,
                true_edges=[("Smoking", "Lung_Cancer")],
            )
        finally:
            app_pipeline.precision_recall_aupr = original_aupr
            app_pipeline.shd = original_shd

        self.assertEqual(metrics["aupr"], 0.42)
        self.assertEqual(metrics["shd"], 3.0)
        self.assertEqual(
            observed_calls,
            [
                ("aupr", ["Lung_Cancer", "Smoking"], ["Lung_Cancer", "Smoking"]),
                ("shd", ["Lung_Cancer", "Smoking"], ["Lung_Cancer", "Smoking"]),
            ],
        )

    def test_run_algorithm_simulation_uses_post_hoc_constraints_for_ges(self):
        def empty_runner(_data_matrix, variable_names, prior_knowledge):
            self.assertIsNone(prior_knowledge)
            graph = nx.DiGraph()
            graph.add_nodes_from(variable_names)
            return graph

        result = run_algorithm_simulation(
            algorithm_name="GES",
            data_matrix=[[0, 1], [1, 0]],
            variable_names=["Smoking", "Lung_Cancer"],
            prior_knowledge=PriorKnowledge(
                required_edges=[("Smoking", "Lung_Cancer")],
                forbidden_edges=[],
            ),
            true_edges=[("Smoking", "Lung_Cancer")],
            runners={"GES": empty_runner},
        )

        self.assertEqual(list(result["baseline_graph"].edges()), [])
        self.assertEqual(
            list(result["constrained_graph"].edges()),
            [("Smoking", "Lung_Cancer")],
        )
        self.assertEqual(result["constraint_mode"], "post_hoc_direct_edge_constraints")
        self.assertEqual(
            result["baseline_metrics"],
            {
                "precision": 0,
                "recall": 0.0,
                "f1": 0,
                "aupr": 0.625,
                "shd": 1.0,
            },
        )
        self.assertEqual(
            result["constrained_metrics"],
            {
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "aupr": 1.0,
                "shd": 0.0,
            },
        )

    def test_run_algorithm_simulation_removes_forbidden_edges_for_ges(self):
        def runner_with_forbidden_edge(_data_matrix, variable_names, prior_knowledge):
            self.assertIsNone(prior_knowledge)
            graph = nx.DiGraph()
            graph.add_nodes_from(variable_names)
            graph.add_edge("Car_Accident", "Smoking")
            return graph

        result = run_algorithm_simulation(
            algorithm_name="GES",
            data_matrix=[[0, 1], [1, 0]],
            variable_names=["Car_Accident", "Smoking"],
            prior_knowledge=PriorKnowledge(
                required_edges=[],
                forbidden_edges=[("Car_Accident", "Smoking")],
            ),
            true_edges=[],
            runners={"GES": runner_with_forbidden_edge},
        )

        self.assertEqual(
            list(result["baseline_graph"].edges()),
            [("Car_Accident", "Smoking")],
        )
        self.assertEqual(list(result["constrained_graph"].edges()), [])

    def test_run_algorithm_simulation_passes_native_prior_knowledge_to_pc(self):
        prior_knowledge = PriorKnowledge(
            required_edges=[("Smoking", "Lung_Cancer")],
            forbidden_edges=[],
        )
        runner_calls = []

        def pc_runner(_data_matrix, variable_names, runner_prior_knowledge):
            runner_calls.append(runner_prior_knowledge)
            graph = nx.DiGraph()
            graph.add_nodes_from(variable_names)

            if runner_prior_knowledge is not None:
                graph.add_edge("Smoking", "Lung_Cancer")

            return graph

        result = run_algorithm_simulation(
            algorithm_name="PC",
            data_matrix=[[0, 1], [1, 0]],
            variable_names=["Smoking", "Lung_Cancer"],
            prior_knowledge=prior_knowledge,
            true_edges=[("Smoking", "Lung_Cancer")],
            runners={"PC": pc_runner},
        )

        self.assertEqual(runner_calls, [None, prior_knowledge])
        self.assertEqual(list(result["baseline_graph"].edges()), [])
        self.assertEqual(
            list(result["constrained_graph"].edges()),
            [("Smoking", "Lung_Cancer")],
        )
        self.assertEqual(result["constraint_mode"], "native_pc_background_knowledge")

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

    def test_build_relations_frame_includes_relation_type(self):
        frame = build_relations_frame(
            [
                {
                    "cause": "Car_Accident",
                    "effect": "Smoking",
                    "confidence": 0.8765,
                    "relation_type": "forbidden",
                },
                {
                    "cause": "Smoking",
                    "effect": "Lung_Cancer",
                    "confidence": 0.95,
                },
            ]
        )

        self.assertEqual(
            list(frame.columns),
            ["relation_type", "cause", "effect", "confidence"],
        )
        self.assertEqual(frame.iloc[0].to_dict()["relation_type"], "forbidden")
        self.assertEqual(frame.iloc[1].to_dict()["relation_type"], "required")
        self.assertEqual(frame.iloc[0].to_dict()["confidence"], 0.876)


if __name__ == "__main__":
    unittest.main()
