"""Run one interventional Sachs ablation cell (PR6): PC/GES/LiNGAM naive + GIES naive/interventional."""

from __future__ import annotations

import random
from typing import Any, Literal

import networkx as nx
import numpy as np

from causal_discovery.run_ges import run_ges
from causal_discovery.run_gies import run_gies, run_gies_naive_pooled
from causal_discovery.run_lingam import run_lingam
from causal_discovery.run_pc import consume_pc_post_hoc_required_added, run_pc
from constraints.constraint_builder import (
    ClaimRecord,
    ConstraintBuilder,
    PriorKnowledge,
)
from evaluation.harness import evaluate
from experiments.run_condition import (
    _derive_condition_label,
    _json_safe_metrics,
    _predicted_edges_list,
    make_constraint_builder,
)
from utils.graph_utils import apply_post_hoc_edits

_VALID_PRIORS_SOURCES = frozenset(
    {
        "freetext_llm",
        "none",
        "omnipath_all",
        "omnipath_reactome_only",
        "oracle",
        "reactome_llm",
        "reactome_llm_with_freetext_fallback",
    }
)
_VALID_ALGORITHMS = frozenset({"PC", "GES", "LiNGAM", "GIES"})
_InterventionStrategy = Literal["naive", "gies"]


def _prior_knowledge_or_none(
    builder: ConstraintBuilder | None,
) -> PriorKnowledge | None:
    if builder is None:
        return None
    return builder.to_prior_knowledge()


def _validate(
    *,
    priors: list[ClaimRecord] | None,
    priors_source: str,
    algorithm: str,
    intervention_strategy: str,
    threshold: float | None,
) -> None:
    if priors_source not in _VALID_PRIORS_SOURCES:
        raise ValueError(f"Invalid priors_source: {priors_source!r}")
    if algorithm not in _VALID_ALGORITHMS:
        raise ValueError(f"Invalid algorithm: {algorithm!r}")
    if intervention_strategy not in {"naive", "gies"}:
        raise ValueError(f"Invalid intervention_strategy: {intervention_strategy!r}")
    if algorithm != "GIES" and intervention_strategy != "naive":
        raise ValueError(
            "PC/GES/LiNGAM interventional runs use intervention_strategy='naive' only"
        )
    if algorithm == "GIES" and intervention_strategy not in {"naive", "gies"}:
        raise ValueError("GIES requires intervention_strategy 'naive' or 'gies'")

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


def run_interventional_condition(
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
    gold_version: str,
    intervention_strategy: _InterventionStrategy,
    block_sizes: list[int],
    gies_intervention_targets: list[list[int]],
    lingam_prior_mode: str | None = None,
) -> dict[str, Any]:
    _validate(
        priors=priors,
        priors_source=priors_source,
        algorithm=algorithm,
        intervention_strategy=intervention_strategy,
        threshold=threshold,
    )

    condition = _derive_condition_label(priors_source, threshold)

    out: dict[str, Any] = {
        "dataset": dataset_name,
        "condition": condition,
        "algorithm": algorithm,
        "gold_version": gold_version,
        "seed": seed,
        "threshold": threshold,
        "priors_source": priors_source,
        "intervention_strategy": intervention_strategy,
        "status": "ok",
        "error": None,
        "metrics": None,
        "predicted_edges": [],
        "constraint_summary": None,
        "dropped_due_to_cycle": [],
        "pc_post_hoc_required_added": 0,
        "pc_post_hoc_dropped_due_to_cycle": [],
    }
    if algorithm == "LiNGAM" and priors_source != "none":
        from experiments.run_condition import _DEFAULT_LINGAM_PRIOR_MODE

        eff_lingam = lingam_prior_mode or _DEFAULT_LINGAM_PRIOR_MODE
        out["lingam_prior_mode"] = eff_lingam
    elif lingam_prior_mode is not None:
        out["lingam_prior_mode"] = lingam_prior_mode

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
        pc_required_added = 0

        np.random.seed(seed)
        random.seed(seed)

        pk = _prior_knowledge_or_none(builder)

        if algorithm == "PC":
            predicted, pc_dropped = run_pc(data, variable_names, pk)
            pc_required_added = consume_pc_post_hoc_required_added(predicted)
            dropped = pc_dropped
        elif algorithm == "LiNGAM":
            if builder is None:
                predicted = run_lingam(data, variable_names, None)
            else:
                from experiments.run_condition import (
                    _DEFAULT_LINGAM_PRIOR_MODE,
                    _build_lingam_sweep_matrix,
                )

                eff_mode = lingam_prior_mode or _DEFAULT_LINGAM_PRIOR_MODE
                if eff_mode == "post_hoc":
                    predicted = run_lingam(data, variable_names, None)
                    add, forbid = builder.build_ges_post_hoc_edits()
                    predicted, dropped = apply_post_hoc_edits(predicted, add, forbid)
                else:
                    matrix = _build_lingam_sweep_matrix(builder, eff_mode)
                    predicted = run_lingam(
                        data,
                        variable_names,
                        prior_matrix=matrix,
                        apply_prior_knowledge_softly=True,
                    )
        elif algorithm == "GES":
            predicted = run_ges(data, variable_names, None)
            if priors_source != "none" and builder is not None:
                add, forbid = builder.build_ges_post_hoc_edits()
                predicted, dropped = apply_post_hoc_edits(predicted, add, forbid)
        elif algorithm == "GIES":
            if intervention_strategy == "naive":
                predicted = run_gies_naive_pooled(data, variable_names, None)
            else:
                predicted = run_gies(
                    data,
                    variable_names,
                    block_sizes=block_sizes,
                    intervention_targets=gies_intervention_targets,
                    prior_knowledge=None,
                )
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
        out["pc_post_hoc_required_added"] = int(pc_required_added)
        out["pc_post_hoc_dropped_due_to_cycle"] = (
            [[pair[0], pair[1]] for pair in sorted(dropped, key=lambda t: (t[0], t[1]))]
            if algorithm == "PC"
            else []
        )

    except Exception as exc:  # noqa: BLE001 — sweep must not crash the runner
        out["status"] = "failed"
        out["error"] = str(exc)
        out["metrics"] = None
        out["predicted_edges"] = []
        out["constraint_summary"] = None
        out["dropped_due_to_cycle"] = []
        out["pc_post_hoc_required_added"] = 0
        out["pc_post_hoc_dropped_due_to_cycle"] = []

    return out
