"""CPDAG-aware precision/recall/F1 and SHD for PC/GES Markov-equivalence outputs.

PC and GES return completed Partially Directed Acyclic Graphs (CPDAGs). Evaluating
them with purely directed arc overlap penalises correctly recovered skeleton edges
that remain unoriented. Following the standard treatment (Chickering, 1995; Tsamardinos
& Brown, 2008), an **undirected** predicted adjacency that aligns with a **gold
directed** edge receives fractional credit: **0.5** toward both precision and recall
for that edge. A predicted arc in the correct direction scores **1.0**; a predicted
arc in the wrong direction scores **0** (same as directed-only metrics).

**SHD-CPDAG**: for each unordered variable pair we charge **1** if the skeleton
membership or orientation class disagrees between predicted CPDAG and gold DAG.
In particular, a gold directed edge paired with a predicted undirected edge counts
**1** error (not **2**, as a naive symmetric adjacency difference on a
``(u,v)+(v,u)`` DiGraph encoding would).

References
----------
Chickering, D. M. (1995). *Learning Equivalence Classes of Bayesian-Network Structures.*
Tsamardinos, I., & Brown, L. E. (2008). *Towards Empirical Evaluation of Time-Efficient
Boolean Network Inference.*

Encoding
--------
Predicted CPDAGs are represented as a :class:`networkx.DiGraph` with an undirected
edge ``u --- v`` stored as both ``(u, v)`` and ``(v, u)``.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

from evaluation.metrics import f1_score


def _arc_set_from_inputs(
    predicted_edges: Iterable[tuple[str, str]],
    predicted_undirected_edges: Iterable[tuple[str, str]],
) -> set[tuple[str, str]]:
    arcs: set[tuple[str, str]] = {(str(u), str(v)) for u, v in predicted_edges}
    for u, v in predicted_undirected_edges:
        a, b = str(u), str(v)
        arcs.add((a, b))
        arcs.add((b, a))
    return arcs


def _normalize_dir_and_undirected(
    arc_set: set[tuple[str, str]],
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Split arcs into directed-only and undirected (mutual) components."""
    directed_only: set[tuple[str, str]] = set()
    undirected: set[tuple[str, str]] = set()
    for u, v in arc_set:
        if (v, u) in arc_set:
            undirected.add(tuple(sorted((u, v))))
        else:
            directed_only.add((u, v))
    return directed_only, undirected


def _normalize_predicted_inputs(
    predicted_edges: Iterable[tuple[str, str]],
    predicted_undirected_edges: Iterable[tuple[str, str]],
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Merge edge lists; mutual reverse arcs in ``predicted_edges`` become undirected."""
    return _normalize_dir_and_undirected(
        _arc_set_from_inputs(predicted_edges, predicted_undirected_edges)
    )


def cpdag_precision_recall_f1(
    predicted_edges: Iterable[tuple[str, str]],
    predicted_undirected_edges: Iterable[tuple[str, str]],
    true_directed_edges: Iterable[tuple[str, str]],
) -> dict[str, float]:
    """Return ``cpdag_precision``, ``cpdag_recall``, ``cpdag_f1`` in ``[0, 1]``."""
    pred_dir, pred_und = _normalize_predicted_inputs(
        predicted_edges, predicted_undirected_edges
    )
    true_set = {(str(u), str(v)) for u, v in true_directed_edges}

    denom_p = len(pred_dir) + len(pred_und)
    if denom_p == 0:
        p = 0.0
    else:
        num_p = 0.0
        for u, v in pred_dir:
            num_p += 1.0 if (u, v) in true_set else 0.0
        for u, v in pred_und:
            num_p += 0.5 if (u, v) in true_set or (v, u) in true_set else 0.0
        p = num_p / denom_p

    denom_r = len(true_set)
    if denom_r == 0:
        r = 0.0
    else:
        num_r = 0.0
        for u, v in true_set:
            if (u, v) in pred_dir:
                num_r += 1.0
            elif (v, u) in pred_dir:
                num_r += 0.0
            elif tuple(sorted((u, v))) in pred_und:
                num_r += 0.5
            else:
                num_r += 0.0
        r = num_r / denom_r

    return {
        "cpdag_precision": float(p),
        "cpdag_recall": float(r),
        "cpdag_f1": float(f1_score(p, r)),
    }


def shd_cpdag(
    predicted_directed: Iterable[tuple[str, str]],
    predicted_undirected: Iterable[tuple[str, str]],
    true_directed: Iterable[tuple[str, str]],
) -> int:
    """Structural Hamming distance between a predicted CPDAG and a gold DAG."""
    pred_dir, pred_und = _normalize_predicted_inputs(
        predicted_directed, predicted_undirected
    )
    true_set = {(str(u), str(v)) for u, v in true_directed}

    nodes: set[str] = set()
    for u, v in true_set:
        nodes.update((u, v))
    for u, v in pred_dir:
        nodes.update((u, v))
    for u, v in pred_und:
        nodes.update((u, v))

    total = 0
    ordered_nodes = sorted(nodes)
    for a, b in combinations(ordered_nodes, 2):
        true_edge: tuple[str, str] | None = None
        if (a, b) in true_set:
            true_edge = (a, b)
        elif (b, a) in true_set:
            true_edge = (b, a)

        und_key = tuple(sorted((a, b)))
        pred_state: str
        if und_key in pred_und:
            pred_state = "undir"
        elif (a, b) in pred_dir:
            pred_state = f"dir:{a}->{b}"
        elif (b, a) in pred_dir:
            pred_state = f"dir:{b}->{a}"
        else:
            pred_state = "none"

        if true_edge is None:
            total += 0 if pred_state == "none" else 1
            continue
        if pred_state == "none":
            total += 1
            continue
        tu, tv = true_edge
        if pred_state == "undir":
            total += 1
            continue
        if pred_state == f"dir:{tu}->{tv}":
            total += 0
        else:
            total += 1
    return int(total)


def split_cpdag_edges(
    predicted_edges: Iterable[tuple[str, str]],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split a DiGraph edge list into (directed_only, undirected_pairs).

    Undirected CPDAG edges are those with both orientations present; each undirected
    pair is returned once as ``(min, max)``-sorted endpoints.
    """
    pred_dir, pred_und = _normalize_predicted_inputs(predicted_edges, ())
    return sorted(pred_dir), sorted(pred_und)
