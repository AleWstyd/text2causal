"""C-LLM-only ablation: emit a cycle-broken DAG from cached LLM priors.

Reads ``experiments/causal_priors_sachs.json`` and produces a single
predicted DAG with no causal-discovery algorithm involved. The DAG
contains every variable in ``variable_names`` as a node (even if some are
isolated), so downstream evaluation against ground truth is well-defined.

Algorithm (pinned in ``docs/step_04_causal_reasoning.md`` step 7):
1. Filter priors to ``confidence >= threshold`` AND
   ``constraint_type in {hard_required, hard_forbidden_reverse, soft_prior}``.
2. Sort the kept claims by confidence descending (ties broken by
   ``(cause, effect)`` lexically so the DAG is deterministic).
3. Initialise an empty ``nx.DiGraph`` with all ``variable_names`` as nodes.
4. For each claim in order, add the directed ``cause → effect`` edge. If
   the addition closes a cycle, remove it and log the dropped edge under
   ``dropped_due_to_cycle: [...]``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

DEFAULT_CONFIDENCE_THRESHOLD: float = 0.7

KEPT_CONSTRAINT_TYPES: frozenset[str] = frozenset(
    {"hard_required", "hard_forbidden_reverse", "soft_prior"}
)


@dataclass
class DagBuildResult:
    """Internal aggregate returned by :func:`predict_dag_from_priors_with_meta`."""

    graph: nx.DiGraph
    threshold: float
    n_claims_in_priors: int
    n_after_threshold: int
    n_after_constraint_filter: int
    claims_kept: list[dict[str, Any]]
    dropped_due_to_cycle: list[dict[str, Any]] = field(default_factory=list)
    source_priors_hash: str | None = None
    variable_names: list[str] = field(default_factory=list)


def _canonical_priors_hash(priors: dict[str, Any]) -> str:
    """SHA-256 of canonical-form serialisation of the source priors blob."""

    canon = json.dumps(priors, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _select_claims(
    priors: dict[str, Any],
    threshold: float,
) -> tuple[list[dict[str, Any]], int]:
    """Return claims passing the threshold + constraint-type filter."""

    raw_pairs = priors.get("pairs") or []
    if not isinstance(raw_pairs, list):
        return [], 0

    above_threshold = [
        p
        for p in raw_pairs
        if isinstance(p, dict)
        and isinstance(p.get("confidence"), (int, float))
        and float(p["confidence"]) >= threshold
        and p.get("cause")
        and p.get("effect")
        and p["cause"] != "unknown"
        and p["effect"] != "unknown"
    ]

    kept = [
        p for p in above_threshold if p.get("constraint_type") in KEPT_CONSTRAINT_TYPES
    ]
    return kept, len(above_threshold)


def predict_dag_from_priors_with_meta(
    priors: dict[str, Any],
    variable_names: list[str],
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> DagBuildResult:
    """Build a cycle-broken DAG plus metadata in one pass.

    ``priors`` is the *parsed* JSON blob from
    ``experiments/causal_priors_sachs.json`` (matching the schema in the
    Step 4 spec). ``variable_names`` is the canonical Sachs column list;
    every name appears as a node in the returned ``DiGraph`` even if no
    edge survives.
    """

    graph: nx.DiGraph = nx.DiGraph()
    graph.add_nodes_from(variable_names)

    selected, n_above_threshold = _select_claims(priors, confidence_threshold)

    selected.sort(
        key=lambda p: (
            -float(p["confidence"]),
            str(p.get("cause", "")),
            str(p.get("effect", "")),
        )
    )

    valid_nodes = set(variable_names)
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    for claim in selected:
        cause = claim["cause"]
        effect = claim["effect"]
        if cause not in valid_nodes or effect not in valid_nodes:
            dropped.append(
                {
                    "cause": cause,
                    "effect": effect,
                    "confidence": float(claim.get("confidence", 0.0)),
                    "reason": "node_not_in_variable_names",
                }
            )
            continue
        if graph.has_edge(cause, effect):
            continue
        graph.add_edge(cause, effect)
        if not nx.is_directed_acyclic_graph(graph):
            graph.remove_edge(cause, effect)
            dropped.append(
                {
                    "cause": cause,
                    "effect": effect,
                    "confidence": float(claim.get("confidence", 0.0)),
                    "reason": "would_close_cycle",
                }
            )
            continue
        kept.append(
            {
                "cause": cause,
                "effect": effect,
                "confidence": float(claim.get("confidence", 0.0)),
                "constraint_type": claim.get("constraint_type"),
            }
        )

    return DagBuildResult(
        graph=graph,
        threshold=confidence_threshold,
        n_claims_in_priors=len(priors.get("pairs") or []),
        n_after_threshold=n_above_threshold,
        n_after_constraint_filter=len(selected),
        claims_kept=kept,
        dropped_due_to_cycle=dropped,
        source_priors_hash=_canonical_priors_hash(priors),
        variable_names=list(variable_names),
    )


def predict_dag_from_priors(
    priors_path: Path,
    variable_names: list[str],
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> nx.DiGraph:
    """Public entry point per ``docs/step_04_causal_reasoning.md`` step 7."""

    priors = json.loads(Path(priors_path).read_text(encoding="utf-8"))
    result = predict_dag_from_priors_with_meta(
        priors, variable_names, confidence_threshold=confidence_threshold
    )
    return result.graph


def build_meta(result: DagBuildResult) -> dict[str, Any]:
    """Return the JSON-serialisable sidecar metadata for the DAG artefact."""

    return {
        "schema_version": "step04.dag_llm_only.v1",
        "threshold": result.threshold,
        "n_claims_in_priors": result.n_claims_in_priors,
        "n_above_threshold": result.n_after_threshold,
        "n_after_constraint_filter": result.n_after_constraint_filter,
        "n_claims_kept": len(result.claims_kept),
        "n_dropped_due_to_cycle": sum(
            1 for d in result.dropped_due_to_cycle if d["reason"] == "would_close_cycle"
        ),
        "n_dropped_other": sum(
            1 for d in result.dropped_due_to_cycle if d["reason"] != "would_close_cycle"
        ),
        "node_count": result.graph.number_of_nodes(),
        "edge_count": result.graph.number_of_edges(),
        "is_acyclic": nx.is_directed_acyclic_graph(result.graph),
        "claims_kept": result.claims_kept,
        "dropped_due_to_cycle": result.dropped_due_to_cycle,
        "source_priors_hash": result.source_priors_hash,
        "variable_names": result.variable_names,
    }
