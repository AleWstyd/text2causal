"""Run one (dataset × priors_source × algorithm × threshold × seed) discovery cell (Step 5 Phase 3)."""

from __future__ import annotations

import random
from typing import Any

import networkx as nx
import numpy as np

from causal_discovery.run_ges import run_ges
from causal_discovery.run_lingam import run_lingam
from causal_discovery.run_pc import run_pc
from constraints.constraint_builder import (
    ClaimRecord,
    ConstraintBuilder,
    PriorKnowledge,
)
from evaluation.harness import evaluate
from utils.graph_utils import apply_post_hoc_edits

_VALID_PRIORS_SOURCES = frozenset(
    {
        "freetext_llm",
        "none",
        "omnipath_all",
        "omnipath_reactome_only",
        "oracle",
        "reactome_llm",
    }
)
_VALID_ALGORITHMS = frozenset({"PC", "GES", "LiNGAM"})


def _derive_condition_label(priors_source: str, threshold: float | None) -> str:
    if priors_source == "none":
        return "C0"
    if priors_source == "omnipath_all":
        return "C0.5"
    if priors_source == "omnipath_reactome_only":
        return "C0.5_reactome_only"
    if priors_source == "oracle":
        return "C5"
    if priors_source == "reactome_llm":
        if threshold == 0.9:
            return "C2"
        if threshold == 0.7:
            return "C3"
        if threshold == 0.6:
            return "C4"
        return f"C_llm_t{threshold}"
    if priors_source == "freetext_llm":
        if threshold == 0.7:
            return "C1"
        return f"C_freetext_t{threshold}"
    raise ValueError(f"Unknown priors_source for condition label: {priors_source!r}")


def _oracle_priors(
    true_graph: nx.DiGraph, variable_names: list[str]
) -> list[ClaimRecord]:
    known = set(variable_names)
    for node in true_graph.nodes():
        if node not in known:
            raise ValueError(
                f"True graph node {node!r} is not in variable_names "
                f"(extra in graph vs variables)"
            )
    claims: list[ClaimRecord] = []
    for cause, effect in true_graph.edges():
        claims.append(
            ClaimRecord(
                var_a=cause,
                var_b=effect,
                cause=cause,
                effect=effect,
                confidence=1.0,
                constraint_type="hard_required",
                source="oracle",
            )
        )
    return claims


def _validate_run_condition_inputs(
    *,
    priors: list[ClaimRecord] | None,
    priors_source: str,
    algorithm: str,
    threshold: float | None,
) -> None:
    if priors_source not in _VALID_PRIORS_SOURCES:
        raise ValueError(f"Invalid priors_source: {priors_source!r}")
    if algorithm not in _VALID_ALGORITHMS:
        raise ValueError(f"Invalid algorithm: {algorithm!r}")

    if priors_source == "none":
        if priors is not None:
            raise ValueError("priors must be None when priors_source is 'none'")
        if threshold is not None:
            raise ValueError("threshold must be None when priors_source is 'none'")
        return

    if priors_source == "oracle":
        return

    if priors is None:
        raise ValueError(
            f"priors must not be None when priors_source is {priors_source!r}"
        )
    if threshold is None:
        raise ValueError(
            f"threshold must not be None when priors_source is {priors_source!r}"
        )


def make_constraint_builder(
    *,
    priors_source: str,
    priors: list[ClaimRecord] | None,
    variable_names: list[str],
    threshold: float | None,
    true_graph: nx.DiGraph,
) -> ConstraintBuilder | None:
    """Return a :class:`ConstraintBuilder`, or ``None`` for the unconstrained baseline (C0)."""
    if priors_source == "none":
        return None

    if priors_source == "oracle":
        eff_threshold = 0.5 if threshold is None else threshold
        op = _oracle_priors(true_graph, variable_names)
        return ConstraintBuilder(op, variable_names, eff_threshold)

    assert priors is not None
    assert threshold is not None
    return ConstraintBuilder(priors, variable_names, threshold)


def _prior_knowledge_or_none(
    builder: ConstraintBuilder | None,
) -> PriorKnowledge | None:
    if builder is None:
        return None
    return builder.to_prior_knowledge()


def _predicted_edges_list(graph: nx.DiGraph) -> list[list[str]]:
    edges = [(str(u), str(v)) for u, v in graph.edges()]
    edges.sort(key=lambda t: (t[0], t[1]))
    return [list(pair) for pair in edges]


def _json_safe_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in metrics.items()}


def run_condition(
    *,
    dataset_name: str,
    data: np.ndarray,
    variable_names: list[str],
    true_graph: nx.DiGraph,
    priors: list[ClaimRecord] | None,
    priors_source: str,
    algorithm: str,
    threshold: float | None,
    seed: int,
) -> dict[str, Any]:
    _validate_run_condition_inputs(
        priors=priors,
        priors_source=priors_source,
        algorithm=algorithm,
        threshold=threshold,
    )

    condition = _derive_condition_label(priors_source, threshold)

    out: dict[str, Any] = {
        "dataset": dataset_name,
        "condition": condition,
        "algorithm": algorithm,
        "seed": seed,
        "threshold": threshold,
        "priors_source": priors_source,
        "status": "ok",
        "error": None,
        "metrics": None,
        "predicted_edges": [],
        "constraint_summary": None,
        "dropped_due_to_cycle": [],
    }

    try:
        builder = make_constraint_builder(
            priors_source=priors_source,
            priors=priors,
            variable_names=variable_names,
            threshold=threshold,
            true_graph=true_graph,
        )

        constraint_summary: dict[str, int] | None = None
        if builder is not None:
            constraint_summary = builder.summary()

        dropped: list[tuple[str, str]] = []

        np.random.seed(seed)
        random.seed(seed)

        if algorithm == "PC":
            pk = _prior_knowledge_or_none(builder)
            predicted = run_pc(data, variable_names, pk)
        elif algorithm == "LiNGAM":
            pk = _prior_knowledge_or_none(builder)
            predicted = run_lingam(data, variable_names, pk)
        elif algorithm == "GES":
            predicted = run_ges(data, variable_names, None)
            if priors_source != "none" and builder is not None:
                add, forbid = builder.build_ges_post_hoc_edits()
                predicted, dropped = apply_post_hoc_edits(predicted, add, forbid)
        else:
            raise ValueError(f"Invalid algorithm: {algorithm!r}")

        metrics = evaluate(predicted, true_graph)
        out["metrics"] = _json_safe_metrics(metrics)
        out["predicted_edges"] = _predicted_edges_list(predicted)
        out["constraint_summary"] = constraint_summary
        out["dropped_due_to_cycle"] = [
            [pair[0], pair[1]] for pair in sorted(dropped, key=lambda t: (t[0], t[1]))
        ]

    except Exception as exc:  # noqa: BLE001 — sweep must not crash the runner
        out["status"] = "failed"
        out["error"] = str(exc)
        out["metrics"] = None
        out["predicted_edges"] = []
        out["constraint_summary"] = None
        out["dropped_due_to_cycle"] = []

    return out
