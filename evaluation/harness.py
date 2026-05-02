from __future__ import annotations

import networkx as nx

from evaluation.metrics import f1_score, precision, precision_recall_aupr, recall, shd


def evaluate(predicted: nx.DiGraph, true: nx.DiGraph) -> dict[str, float]:
    """Evaluate a predicted graph against a true graph with aligned node sets."""
    all_nodes = sorted(set(predicted.nodes()) | set(true.nodes()))
    predicted_graph = predicted.copy()
    true_graph = true.copy()
    predicted_graph.add_nodes_from(all_nodes)
    true_graph.add_nodes_from(all_nodes)

    metric_precision = precision(
        list(predicted_graph.edges()), list(true_graph.edges())
    )
    metric_recall = recall(list(predicted_graph.edges()), list(true_graph.edges()))

    return {
        "shd": shd(true_graph, predicted_graph),
        "aupr": precision_recall_aupr(true_graph, predicted_graph),
        "precision": metric_precision,
        "recall": metric_recall,
        "f1": f1_score(metric_precision, metric_recall),
    }
