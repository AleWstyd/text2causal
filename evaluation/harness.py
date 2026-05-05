from __future__ import annotations

from typing import Any

import networkx as nx

from evaluation.cpdag_metrics import (
    cpdag_precision_recall_f1,
    shd_cpdag,
    split_cpdag_edges,
)
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

    pred_dir, pred_und = split_cpdag_edges(pred_e)
    cpdag = cpdag_precision_recall_f1(pred_dir, pred_und, true_e)

    return {
        "shd": shd(true_graph, predicted_graph),
        "aupr": precision_recall_aupr(true_graph, predicted_graph),
        "precision": metric_precision,
        "recall": metric_recall,
        "f1": f1_score(metric_precision, metric_recall),
        "directed_precision": d_precision,
        "directed_recall": d_recall,
        "directed_f1": directed_f1(pred_e, true_e),
        "cpdag_precision": cpdag["cpdag_precision"],
        "cpdag_recall": cpdag["cpdag_recall"],
        "cpdag_f1": cpdag["cpdag_f1"],
        "shd_cpdag": float(shd_cpdag(pred_dir, pred_und, true_e)),
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


def row_ok_metrics_missing_cpdag_f1(row: dict[str, Any]) -> bool:
    if row.get("status") != "ok":
        return False
    m = row.get("metrics")
    if not isinstance(m, dict):
        return True
    return "cpdag_f1" not in m


def predicted_edges_row_has_mutual_arc_encoding(row: dict[str, Any]) -> bool:
    """True when some unordered pair appears as both orientations in ``predicted_edges``."""
    raw = row.get("predicted_edges") or []
    arcs = {(str(e[0]), str(e[1])) for e in raw if len(e) == 2}
    return any((b, a) in arcs for a, b in arcs if a != b)


def _true_graph_for_ablation_row(
    row: dict[str, Any],
    *,
    default_true_graph: nx.DiGraph,
    true_graph_by_gold_version: dict[str, nx.DiGraph] | None,
) -> nx.DiGraph:
    if true_graph_by_gold_version is None:
        return default_true_graph
    gv = str(row.get("gold_version") or "original")
    return true_graph_by_gold_version.get(gv, default_true_graph)


def recompute_metrics_from_predicted_edges(
    row: dict[str, Any],
    variable_names: list[str],
    true_graph: nx.DiGraph,
) -> None:
    """Update metric keys in ``row["metrics"]`` from a fresh :func:`evaluate` call."""
    if row.get("status") != "ok":
        return
    pred = digraph_from_predicted_edges_row(row, variable_names)
    fresh = evaluate(pred, true_graph)
    m = row.setdefault("metrics", {})
    for key in (
        "directed_precision",
        "directed_recall",
        "directed_f1",
        "cpdag_precision",
        "cpdag_recall",
        "cpdag_f1",
        "shd_cpdag",
    ):
        m[key] = float(fresh[key])


def backfill_directed_metrics_in_results(
    results: list[dict[str, Any]],
    variable_names: list[str],
    true_graph: nx.DiGraph,
    *,
    true_graph_by_gold_version: dict[str, nx.DiGraph] | None = None,
) -> int:
    """Re-evaluate rows missing ``directed_f1``; returns count of updated rows."""
    n = 0
    for row in results:
        if row_ok_metrics_missing_directed_f1(row):
            tg = _true_graph_for_ablation_row(
                row,
                default_true_graph=true_graph,
                true_graph_by_gold_version=true_graph_by_gold_version,
            )
            recompute_metrics_from_predicted_edges(row, variable_names, tg)
            n += 1
    return n


def backfill_cpdag_metrics_in_results(
    results: list[dict[str, Any]],
    variable_names: list[str],
    true_graph: nx.DiGraph,
    *,
    true_graph_by_gold_version: dict[str, nx.DiGraph] | None = None,
) -> int:
    """Add ``cpdag_*`` / ``shd_cpdag`` to ok rows missing them (no algorithm rerun).

    Rows whose ``predicted_edges`` lack the ``(u,v)+(v,u)`` CPDAG encoding are treated
    as fully directed predictions: ``cpdag_*`` copy ``directed_*``, and ``shd_cpdag``
    is computed from the same directed edge list.
    """
    n = 0
    for row in results:
        if not row_ok_metrics_missing_cpdag_f1(row):
            continue
        tg = _true_graph_for_ablation_row(
            row,
            default_true_graph=true_graph,
            true_graph_by_gold_version=true_graph_by_gold_version,
        )
        m = row.setdefault("metrics", {})
        if predicted_edges_row_has_mutual_arc_encoding(row):
            recompute_metrics_from_predicted_edges(row, variable_names, tg)
            n += 1
            continue

        pred_list = [
            (str(e[0]), str(e[1]))
            for e in (row.get("predicted_edges") or [])
            if isinstance(e, (list, tuple)) and len(e) == 2
        ]
        if "directed_f1" not in m:
            recompute_metrics_from_predicted_edges(row, variable_names, tg)
            n += 1
            continue

        m["cpdag_precision"] = float(m["directed_precision"])
        m["cpdag_recall"] = float(m["directed_recall"])
        m["cpdag_f1"] = float(m["directed_f1"])
        m["shd_cpdag"] = float(shd_cpdag(pred_list, [], tg.edges()))
        n += 1
    return n
