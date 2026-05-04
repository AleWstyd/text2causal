"""Step 6 Phase 3 — constraint quality metrics for prior sources (no CD algorithms)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from constraints.constraint_builder import ClaimRecord, load_priors
from utils.load_data import load_sachs_dataset

_PRIORS_FILES: tuple[tuple[str, Path], ...] = (
    ("reactome_llm", Path("experiments/causal_priors_sachs.json")),
    (
        "omnipath_reactome_only",
        Path("experiments/floor_priors_sachs_reactome_only.json"),
    ),
    ("omnipath_all", Path("experiments/floor_priors_sachs_all.json")),
    ("freetext_llm", Path("experiments/freetext_priors_sachs.json")),
)


def _llm_coverage_numerator(priors: list[ClaimRecord]) -> int:
    seen: set[tuple[str, str]] = set()
    for c in priors:
        if c.constraint_type in {"unknown", "no_context"}:
            continue
        seen.add((c.var_a, c.var_b))
    return len(seen)


def _omnipath_directed_pair_count_from_json(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_items = data.get("claims") or data.get("pairs") or []
    if not isinstance(raw_items, list):
        return 0
    directed: set[tuple[str, str]] = set()
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        va: str = str(item["var_a"])
        vb: str = str(item["var_b"])
        if item.get("is_directed") is True:
            directed.add((va, vb))
            continue
        ct = str(item.get("constraint_type", ""))
        if ct in {"hard_required", "soft_prior"}:
            cause, effect = str(item.get("cause", "")), str(item.get("effect", ""))
            if cause != "unknown" and effect != "unknown":
                directed.add((va, vb))
    return len(directed)


def _forward_predicted_edges(
    priors: list[ClaimRecord], confidence_threshold: float
) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for c in priors:
        if c.confidence < confidence_threshold:
            continue
        if c.constraint_type not in {"hard_required", "soft_prior"}:
            continue
        if c.cause == "unknown" or c.effect == "unknown":
            continue
        out.add((c.cause, c.effect))
    return out


def _forbidden_predicted_edges(
    priors: list[ClaimRecord], confidence_threshold: float
) -> set[tuple[str, str]]:
    """Edges we forbid: (effect, cause) for each qualifying hard_forbidden_reverse(cause, effect)."""
    out: set[tuple[str, str]] = set()
    for c in priors:
        if c.confidence < confidence_threshold:
            continue
        if c.constraint_type != "hard_forbidden_reverse":
            continue
        if c.cause == "unknown" or c.effect == "unknown":
            continue
        out.add((c.effect, c.cause))
    return out


def _sorted_edge_list(edges: set[tuple[str, str]]) -> list[list[str]]:
    return [list(e) for e in sorted(edges)]


def compute_quality(
    priors: list[ClaimRecord],
    true_edges: set[tuple[str, str]],
    *,
    confidence_threshold: float = 0.7,
    n_total_pairs: int,
    omnipath_coverage_numerator: int | None = None,
) -> dict[str, Any]:
    """Compute precision, recall, hallucination (strict + loose), coverage.

    Forward predictions at ``confidence_threshold`` (inclusive): ``hard_required`` and
    ``soft_prior`` with known cause/effect yield a predicted edge cause → effect.
    ``hard_forbidden_reverse(cause=A, effect=B)`` forbids B → A; it does not add a
    forward prediction. Optional ``omnipath_coverage_numerator`` overrides the LLM-style
    coverage count for OmniPath floor priors (directed-pair count from JSON, including
    ``is_directed`` rows where ``constraint_type`` may still be ``unknown``).

    When there are no forward predictions, ``precision_forward`` and both hallucination
    rates are ``0.0`` so JSON remains clean.
    """
    predicted_forward = _forward_predicted_edges(priors, confidence_threshold)
    predicted_forbidden = _forbidden_predicted_edges(priors, confidence_threshold)

    n_true = len(true_edges)
    n_fwd = len(predicted_forward)

    true_positives = {e for e in predicted_forward if e in true_edges}
    false_positives_strict: set[tuple[str, str]] = set()
    false_positives_loose: set[tuple[str, str]] = set()
    for a, b in predicted_forward:
        if (a, b) in true_edges:
            continue
        if (b, a) in true_edges:
            false_positives_strict.add((a, b))
        elif (b, a) not in true_edges:
            false_positives_loose.add((a, b))

    precision_forward = (len(true_positives) / n_fwd) if n_fwd else 0.0
    recall_forward = (len(true_positives) / n_true) if n_true else 0.0
    hallucination_strict = (len(false_positives_strict) / n_fwd) if n_fwd else 0.0
    hallucination_loose = (len(false_positives_loose) / n_fwd) if n_fwd else 0.0

    n_forbidden = len(predicted_forbidden)
    if n_forbidden:
        correct_forbidden = sum(
            1 for edge in predicted_forbidden if edge not in true_edges
        )
        precision_forbidden = correct_forbidden / n_forbidden
    else:
        precision_forbidden = 0.0

    if omnipath_coverage_numerator is not None:
        cov_num = omnipath_coverage_numerator
    else:
        cov_num = _llm_coverage_numerator(priors)
    coverage = cov_num / n_total_pairs if n_total_pairs else 0.0

    return {
        "n_predicted_forward": n_fwd,
        "n_predicted_forbidden": n_forbidden,
        "n_true_edges": n_true,
        "precision_forward": float(precision_forward),
        "recall_forward": float(recall_forward),
        "precision_forbidden": float(precision_forbidden),
        "hallucination_strict": float(hallucination_strict),
        "hallucination_loose": float(hallucination_loose),
        "coverage": float(coverage),
        "true_positives": _sorted_edge_list(true_positives),
        "false_positives_strict": _sorted_edge_list(false_positives_strict),
        "false_positives_loose": _sorted_edge_list(false_positives_loose),
    }


def _headline(by_source: dict[str, dict[str, Any]]) -> str:
    parts: list[str] = []
    order = (
        "reactome_llm",
        "omnipath_reactome_only",
        "omnipath_all",
        "freetext_llm",
    )
    for key in order:
        row = by_source[key]
        parts.append(
            f"{key} precision={row['precision_forward']:.2f} "
            f"recall={row['recall_forward']:.2f} "
            f"hallucination_strict={row['hallucination_strict']:.2f} "
            f"coverage={row['coverage']:.2f}"
        )
    return "; ".join(parts)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _data, true_graph = load_sachs_dataset()
    true_edges: set[tuple[str, str]] = {tuple(e) for e in true_graph.edges()}
    n_nodes = true_graph.number_of_nodes()
    n_total_ordered_pairs = n_nodes * (n_nodes - 1)

    by_source: dict[str, dict[str, Any]] = {}
    for source_key, rel_path in _PRIORS_FILES:
        path = repo_root / rel_path
        priors = load_priors(path)
        cov_override: int | None = None
        if source_key.startswith("omnipath"):
            cov_override = _omnipath_directed_pair_count_from_json(path)
        by_source[source_key] = compute_quality(
            priors,
            true_edges,
            confidence_threshold=0.7,
            n_total_pairs=n_total_ordered_pairs,
            omnipath_coverage_numerator=cov_override,
        )

    out: dict[str, Any] = {
        "dataset": "sachs",
        "confidence_threshold": 0.7,
        "n_total_ordered_pairs": n_total_ordered_pairs,
        "n_true_edges": len(true_edges),
        "true_edges": _sorted_edge_list(true_edges),
        "by_source": by_source,
        "headline": _headline(by_source),
    }

    out_path = repo_root / "experiments" / "constraint_quality_sachs.json"
    out_path.write_text(
        json.dumps(out, indent=2) + "\n",
        encoding="utf-8",
    )
    print(out["headline"])


if __name__ == "__main__":
    main()
