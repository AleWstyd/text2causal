"""Step 6.7 — per-edge attribution for Sachs reaction-evidence edges."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Final

ABLATION_PATH: Final[Path] = Path("experiments/ablation_results_sachs.json")
QUALITY_PATH: Final[Path] = Path("experiments/constraint_quality_sachs.json")
OUTPUT_PATH: Final[Path] = Path("experiments/per_edge_attribution_sachs.json")
ALGORITHMS: Final[tuple[str, ...]] = ("PC", "GES", "LiNGAM")
CONDITIONS: Final[tuple[str, ...]] = ("C0", "C2", "C3", "C4")
DEFAULT_PRIOR_CONDITION: Final[str] = "C3"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def _edge_key(edge: list[str] | tuple[str, str]) -> tuple[str, str]:
    if len(edge) != 2:
        raise ValueError(f"Expected 2-node edge, got {edge!r}")
    return (str(edge[0]), str(edge[1]))


def _reaction_true_edges(quality: dict[str, Any]) -> list[tuple[str, str]]:
    cascade = quality.get("edge_quality_cascade")
    if not isinstance(cascade, dict) or "reaction" not in cascade:
        raise ValueError(
            "constraint_quality_sachs.json missing edge_quality_cascade.reaction"
        )
    reaction = cascade["reaction"]
    edges = [_edge_key(edge) for edge in reaction.get("true_positives", [])]
    expected = int(reaction.get("n_true_edges_in_stratum", len(edges)))
    if len(edges) != expected:
        raise ValueError(
            "Reaction stratum true-edge list is incomplete: "
            f"got {len(edges)}, expected {expected}"
        )
    return sorted(edges, key=lambda pair: (pair[0], pair[1]))


def _orientation_state(
    predicted_edges: set[tuple[str, str]], edge: tuple[str, str]
) -> str:
    cause, effect = edge
    forward = (cause, effect) in predicted_edges
    reverse = (effect, cause) in predicted_edges
    if forward and reverse:
        return "bidirectional"
    if forward:
        return "forward"
    if reverse:
        return "reverse"
    return "absent"


def _summarise_rows(
    rows: list[dict[str, Any]],
    *,
    edge: tuple[str, str],
    algorithm: str,
    condition: str,
) -> dict[str, Any]:
    counts = {"forward": 0, "reverse": 0, "bidirectional": 0, "absent": 0}
    considered = 0
    failed = 0

    for row in rows:
        if row.get("condition") != condition or row.get("algorithm") != algorithm:
            continue
        if row.get("status") != "ok":
            failed += 1
            continue
        predicted_edges = {_edge_key(edge_) for edge_ in row.get("predicted_edges", [])}
        counts[_orientation_state(predicted_edges, edge)] += 1
        considered += 1

    forward_rate = counts["forward"] / considered if considered else None
    reverse_rate = counts["reverse"] / considered if considered else None
    absent_rate = counts["absent"] / considered if considered else None
    return {
        "condition": condition,
        "algorithm": algorithm,
        "edge": list(edge),
        "n_ok": considered,
        "n_failed": failed,
        "counts": counts,
        "forward_rate": forward_rate,
        "reverse_rate": reverse_rate,
        "absent_rate": absent_rate,
    }


def build_per_edge_attribution(
    *,
    ablation_path: Path = ABLATION_PATH,
    quality_path: Path = QUALITY_PATH,
) -> dict[str, Any]:
    ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    results = list(ablation.get("results") or [])
    reaction_edges = _reaction_true_edges(quality)

    per_edge: list[dict[str, Any]] = []
    for edge in reaction_edges:
        for algorithm in ALGORITHMS:
            by_condition = [
                _summarise_rows(
                    results,
                    edge=edge,
                    algorithm=algorithm,
                    condition=condition,
                )
                for condition in CONDITIONS
            ]
            c0 = next(row for row in by_condition if row["condition"] == "C0")
            c3 = next(
                row
                for row in by_condition
                if row["condition"] == DEFAULT_PRIOR_CONDITION
            )
            c0_rate = c0["forward_rate"]
            c3_rate = c3["forward_rate"]
            if c0_rate is None or c3_rate is None:
                delta = None
            else:
                delta = c3_rate - c0_rate
            per_edge.append(
                {
                    "edge": list(edge),
                    "algorithm": algorithm,
                    "baseline_condition": "C0",
                    "default_prior_condition": DEFAULT_PRIOR_CONDITION,
                    "default_prior_forward_delta": delta,
                    "conditions": by_condition,
                }
            )

    by_algorithm: dict[str, Any] = {}
    for algorithm in ALGORITHMS:
        alg_rows = [row for row in per_edge if row["algorithm"] == algorithm]
        deltas = [
            float(row["default_prior_forward_delta"])
            for row in alg_rows
            if row["default_prior_forward_delta"] is not None
        ]
        by_algorithm[algorithm] = {
            "n_edges": len(alg_rows),
            "mean_c3_minus_c0_forward_rate": (
                sum(deltas) / len(deltas) if deltas else None
            ),
            "n_edges_improved_by_c3": sum(1 for delta in deltas if delta > 0),
            "n_edges_unchanged_by_c3": sum(1 for delta in deltas if delta == 0),
            "n_edges_worsened_by_c3": sum(1 for delta in deltas if delta < 0),
        }

    return {
        "dataset": "sachs",
        "ablation_path": ablation_path.as_posix(),
        "quality_path": quality_path.as_posix(),
        "reaction_evidence_true_edges": [list(edge) for edge in reaction_edges],
        "conditions_compared": list(CONDITIONS),
        "default_prior_condition": DEFAULT_PRIOR_CONDITION,
        "per_edge": per_edge,
        "by_algorithm": by_algorithm,
    }


def main() -> None:
    payload = build_per_edge_attribution()
    _atomic_write_json(OUTPUT_PATH, payload)
    print(
        "Wrote "
        f"{OUTPUT_PATH} for {len(payload['reaction_evidence_true_edges'])} "
        "reaction-evidence Sachs true edges"
    )
    print(json.dumps(payload["by_algorithm"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
