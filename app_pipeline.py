from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from constraints.constraint_builder import (
    PriorKnowledge,
    build_prior_knowledge,
    validate_prior_knowledge,
)
from evaluation.metrics import f1_score, precision, precision_recall_aupr, recall, shd
from utils.graph_utils import apply_constraints
from utils.load_data import load_lucas_dataset


TRUE_EDGES = [
    ("Smoking", "Yellow_Fingers"),
    ("Anxiety", "Smoking"),
    ("Peer_Pressure", "Smoking"),
    ("Smoking", "Lung_Cancer"),
    ("Lung_Cancer", "Coughing"),
    ("Lung_Cancer", "Fatigue"),
    ("Allergy", "Coughing"),
    ("Coughing", "Fatigue"),
    ("Genetics", "Lung_Cancer"),
    ("Genetics", "Attention_Disorder"),
    ("Fatigue", "Car_Accident"),
    ("Attention_Disorder", "Car_Accident"),
]

AppInputs = dict[str, Any]
MetricDict = dict[str, float]
SimulationResult = dict[str, Any]
RelationExtractor = Callable[[list[str], str], list[dict[str, Any]]]
AlgorithmRunner = Callable[[Any, list[str], PriorKnowledge | None], Any]


def get_algorithm_runners() -> dict[str, AlgorithmRunner]:
    from causal_discovery.run_ges import run_ges
    from causal_discovery.run_lingam import run_lingam
    from causal_discovery.run_pc import run_pc

    return {
        "PC": run_pc,
        "GES": run_ges,
        "LiNGAM": run_lingam,
    }


def get_algorithm_names() -> list[str]:
    return list(get_algorithm_runners().keys())


def load_app_inputs(
    dataset_path: str = "data/lucas",
    background_path: str = "data/lucas/lucas_background.txt",
) -> AppInputs:
    data = load_lucas_dataset(dataset_path)
    background_text = Path(background_path).read_text(encoding="utf-8")

    return {
        "data": data,
        "data_matrix": data.values,
        "variable_names": list(data.columns),
        "background_text": background_text,
    }


def extract_llm_constraints(
    variable_names: list[str],
    background_text: str,
    threshold: float = 0.7,
    extractor: RelationExtractor | None = None,
) -> dict[str, Any]:
    if extractor is None:
        from llm.extract_relations import extract_relations

        extractor = extract_relations

    relations = extractor(variable_names, background_text)
    prior_knowledge = build_prior_knowledge(relations, threshold=threshold)
    validate_prior_knowledge(prior_knowledge, variable_names)

    return {
        "relations": relations,
        "prior_knowledge": prior_knowledge,
    }


def calculate_graph_metrics(
    graph: Any,
    true_edges: list[tuple[str, str]] | None = None,
) -> MetricDict:
    reference_edges = true_edges or TRUE_EDGES
    predicted_edges = list(graph.edges())
    all_nodes = set(graph.nodes())
    for cause, effect in reference_edges:
        all_nodes.add(cause)
        all_nodes.add(effect)

    reference_graph = graph.__class__()
    reference_graph.add_nodes_from(sorted(all_nodes))
    reference_graph.add_edges_from(reference_edges)
    predicted_graph = graph.copy()
    predicted_graph.add_nodes_from(sorted(all_nodes))
    metric_precision = precision(predicted_edges, reference_edges)
    metric_recall = recall(predicted_edges, reference_edges)

    return {
        "precision": metric_precision,
        "recall": metric_recall,
        "f1": f1_score(metric_precision, metric_recall),
        "aupr": precision_recall_aupr(reference_graph, predicted_graph),
        "shd": shd(reference_graph, predicted_graph),
    }


def run_algorithm_simulation(
    algorithm_name: str,
    data_matrix: Any,
    variable_names: list[str],
    prior_knowledge: PriorKnowledge,
    true_edges: list[tuple[str, str]] | None = None,
    runners: dict[str, AlgorithmRunner] | None = None,
) -> SimulationResult:
    available_runners = runners or get_algorithm_runners()

    if algorithm_name not in available_runners:
        available = ", ".join(available_runners)
        raise ValueError(
            f"Unsupported algorithm '{algorithm_name}'. Available: {available}"
        )

    validate_prior_knowledge(prior_knowledge, variable_names)

    baseline_graph = available_runners[algorithm_name](
        data_matrix, variable_names, None
    )

    if algorithm_name == "GES":
        constrained_graph = apply_constraints(baseline_graph.copy(), prior_knowledge)
        constraint_mode = "post_hoc_direct_edge_constraints"
    elif algorithm_name == "PC":
        constrained_graph = available_runners[algorithm_name](
            data_matrix, variable_names, prior_knowledge
        )
        constraint_mode = "native_pc_background_knowledge"
    else:
        constrained_graph = available_runners[algorithm_name](
            data_matrix, variable_names, prior_knowledge
        )
        constraint_mode = "native_direct_lingam_prior_knowledge"

    return {
        "algorithm_name": algorithm_name,
        "baseline_graph": baseline_graph,
        "constrained_graph": constrained_graph,
        "baseline_metrics": calculate_graph_metrics(baseline_graph, true_edges),
        "constrained_metrics": calculate_graph_metrics(constrained_graph, true_edges),
        "prior_knowledge": prior_knowledge,
        "constraint_mode": constraint_mode,
    }
