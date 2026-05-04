from __future__ import annotations

from typing import Any

import networkx as nx

from evaluation.metrics import (
    directed_f1,
    directed_precision,
    directed_recall,
    f1_score,
    precision,
    precision_recall_aupr,
    recall,
    shd,
)


def evaluate(predicted: nx.DiGraph, true: nx.DiGraph) -> dict[str, float]:
    """Evaluate a predicted graph against a true graph with aligned node sets."""
    all_nodes = sorted(set(predicted.nodes()) | set(true.nodes()))
    predicted_graph = predicted.copy()
    true_graph = true.copy()
    predicted_graph.add_nodes_from(all_nodes)
    true_graph.add_nodes_from(all_nodes)

    pred_e = list(predicted_graph.edges())
    true_e = list(true_graph.edges())
    metric_precision = precision(pred_e, true_e)
    metric_recall = recall(pred_e, true_e)
    d_precision = directed_precision(pred_e, true_e)
    d_recall = directed_recall(pred_e, true_e)

    return {
        "shd": shd(true_graph, predicted_graph),
        "aupr": precision_recall_aupr(true_graph, predicted_graph),
        "precision": metric_precision,
        "recall": metric_recall,
        "f1": f1_score(metric_precision, metric_recall),
        "directed_precision": d_precision,
        "directed_recall": d_recall,
        "directed_f1": directed_f1(pred_e, true_e),
    }


def digraph_from_predicted_edges_row(
    row: dict[str, Any], variable_names: list[str]
) -> nx.DiGraph:
    """Reconstruct a predicted graph from a run-condition / ablation result row."""
    g: nx.DiGraph = nx.DiGraph()
    g.add_nodes_from(variable_names)
    for edge in row.get("predicted_edges") or []:
        if len(edge) == 2:
            g.add_edge(str(edge[0]), str(edge[1]))
    return g


def row_ok_metrics_missing_directed_f1(row: dict[str, Any]) -> bool:
    if row.get("status") != "ok":
        return False
    m = row.get("metrics")
    if not isinstance(m, dict):
        return True
    return "directed_f1" not in m


def recompute_metrics_from_predicted_edges(
    row: dict[str, Any],
    variable_names: list[str],
    true_graph: nx.DiGraph,
) -> None:
    """Add directed metric keys to ``row["metrics"]`` from a fresh :func:`evaluate` call."""
    if row.get("status") != "ok":
        return
    pred = digraph_from_predicted_edges_row(row, variable_names)
    fresh = evaluate(pred, true_graph)
    m = row.setdefault("metrics", {})
    for key in ("directed_precision", "directed_recall", "directed_f1"):
        m[key] = float(fresh[key])


def backfill_directed_metrics_in_results(
    results: list[dict[str, Any]],
    variable_names: list[str],
    true_graph: nx.DiGraph,
) -> int:
    """Re-evaluate rows missing ``directed_f1``; returns count of updated rows."""
    n = 0
    for row in results:
        if row_ok_metrics_missing_directed_f1(row):
            recompute_metrics_from_predicted_edges(row, variable_names, true_graph)
            n += 1
    return n
