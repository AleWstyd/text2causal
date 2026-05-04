from __future__ import annotations

import networkx as nx
import numpy as np

try:
    from cdt.metrics import SHD as cdt_shd
    from cdt.metrics import precision_recall as cdt_precision_recall
except ImportError:  # pragma: no cover - exercised only when cdt is unavailable.
    cdt_shd = None
    cdt_precision_recall = None


def _normalize_edges(edges):
    """Normalize edges to undirected form for comparison."""
    return {tuple(sorted([u, v])) for u, v in edges}


def _directed_edges(edges):
    """Directed edge pairs as ``(str, str)`` ordered tuples (no normalisation)."""
    return {(str(u), str(v)) for u, v in edges}


def _retrieve_adjacency_matrix(graph, order_nodes=None, weight=False):
    if isinstance(graph, np.ndarray):
        return graph

    if isinstance(graph, nx.DiGraph):
        node_order = (
            list(order_nodes) if order_nodes is not None else list(graph.nodes())
        )
        if not weight:
            return np.array(
                nx.adjacency_matrix(graph, node_order, weight=None).todense()
            )

        return np.array(nx.adjacency_matrix(graph, node_order).todense())

    raise TypeError("Only networkx.DiGraph and np.ndarray are supported.")


def precision_recall_aupr(target_graph, predicted_graph):
    if cdt_precision_recall is not None:
        aupr, _curve = cdt_precision_recall(target_graph, predicted_graph)
        return float(aupr)

    from sklearn.metrics import auc, precision_recall_curve

    true_labels = _retrieve_adjacency_matrix(target_graph)
    predictions = _retrieve_adjacency_matrix(
        predicted_graph,
        target_graph.nodes() if isinstance(target_graph, nx.DiGraph) else None,
        weight=True,
    )

    precision_values, recall_values, _ = precision_recall_curve(
        true_labels.ravel(), predictions.ravel()
    )
    return float(auc(recall_values, precision_values))


def shd(target_graph, predicted_graph, double_for_anticausal=True):
    if cdt_shd is not None:
        return float(cdt_shd(target_graph, predicted_graph, double_for_anticausal))

    true_labels = _retrieve_adjacency_matrix(target_graph)
    predictions = _retrieve_adjacency_matrix(
        predicted_graph,
        target_graph.nodes() if isinstance(target_graph, nx.DiGraph) else None,
    )
    diff = np.abs(true_labels - predictions)

    if double_for_anticausal:
        return float(np.sum(diff))

    diff = diff + diff.transpose()
    diff[diff > 1] = 1
    return float(np.sum(diff) / 2)


def precision(predicted_edges, true_edges):
    """Skeleton (undirected) precision: edges are compared after sorting endpoints.

    A predicted arc matches a true arc if the unordered pair is the same; the
    reverse orientation counts as a true positive.
    """
    predicted = _normalize_edges(predicted_edges)
    true = _normalize_edges(true_edges)

    if len(predicted) == 0:
        return 0

    correct = predicted.intersection(true)

    return len(correct) / len(predicted)


def recall(predicted_edges, true_edges):
    """Skeleton (undirected) recall: edges are compared after sorting endpoints.

    See :func:`precision` for the orientation-agnostic matching rule.
    """
    predicted = _normalize_edges(predicted_edges)
    true = _normalize_edges(true_edges)

    correct = predicted.intersection(true)

    return len(correct) / len(true)


def directed_precision(predicted_edges, true_edges):
    """Directed precision: ``(u, v)`` must match exactly; no endpoint swapping."""
    predicted = _directed_edges(predicted_edges)
    true = _directed_edges(true_edges)
    if len(predicted) == 0:
        return 0.0
    correct = predicted.intersection(true)
    return len(correct) / len(predicted)


def directed_recall(predicted_edges, true_edges):
    """Directed recall: ``(u, v)`` must match exactly; no endpoint swapping."""
    predicted = _directed_edges(predicted_edges)
    true = _directed_edges(true_edges)
    if len(true) == 0:
        return 0.0
    correct = predicted.intersection(true)
    return len(correct) / len(true)


def directed_f1(predicted_edges, true_edges):
    """Harmonic mean of :func:`directed_precision` and :func:`directed_recall`."""
    return f1_score(
        directed_precision(predicted_edges, true_edges),
        directed_recall(predicted_edges, true_edges),
    )


def f1_score(p, r):

    if (p + r) == 0:
        return 0

    return 2 * (p * r) / (p + r)


def calculate_metrics(cg, true_graph):
    """Calculate evaluation metrics: Precision, Recall, F1-Score."""
    inferred_edges = _normalize_edges(cg["model_edges"])
    true_edges = _normalize_edges(true_graph.edges())

    tp = len(inferred_edges & true_edges)  # True positives
    fp = len(inferred_edges - true_edges)  # False positives
    fn = len(true_edges - inferred_edges)  # False negatives

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    )

    return precision, recall, f1
