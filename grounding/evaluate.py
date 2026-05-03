"""Metrics and failure logging for variable grounding vs a gold fixture."""

from __future__ import annotations

from typing import Any, Literal

from grounding.ground import CHEBI_PAIR_PIP2, CHEBI_PAIR_PIP3, Grounding

PHOSPHOINOSITIDE_DUAL: dict[str, frozenset[str]] = {
    "PIP2": CHEBI_PAIR_PIP2,
    "PIP3": CHEBI_PAIR_PIP3,
}


def _norm_id_set(ids: list[str]) -> set[str]:
    return set(ids)


FailureReason = Literal[
    "missing_gold_id",
    "extra_predicted_id",
    "unvalidated",
    "kind_mismatch",
    "missing_chebi_form",
]


def evaluate_grounding(
    predicted: dict[str, Grounding],
    gold: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compare ``Grounding`` predictions to a gold fixture (per-column dicts)."""

    columns = sorted(set(predicted) | set(gold))
    failures: list[dict[str, Any]] = []
    recalls: list[float] = []
    precs: list[float] = []
    jaccards: list[float] = []
    kinds_ok = 0
    canon_match = 0
    n_validated = 0

    for col in columns:
        pred = predicted.get(col)
        gold_entry = gold.get(col)
        if gold_entry is None or pred is None:
            failures.append(
                {
                    "column": col,
                    "reason": "missing_gold_id"
                    if gold_entry is None
                    else "extra_predicted_id",
                    "expected": gold_entry,
                    "got": None if pred is None else pred.as_dict(),
                },
            )
            recalls.append(0.0)
            precs.append(0.0)
            jaccards.append(0.0)
            continue

        g_kind_raw = gold_entry.get("kind")
        if isinstance(g_kind_raw, str):
            g_kind: str | None = g_kind_raw.lower()
        else:
            g_kind = None

        ids_raw = gold_entry.get("ids", [])
        if not isinstance(ids_raw, list):
            gold_ids_list: list[str] = []
        else:
            gold_ids_list = [x for x in ids_raw if isinstance(x, str)]
        gold_set = _norm_id_set(gold_ids_list)
        pred_set = _norm_id_set(pred.ids)
        canon_gold = ""
        cg = gold_entry.get("canonical_name")
        if isinstance(cg, str):
            canon_gold = cg

        intersect = gold_set & pred_set
        if g_kind == pred.kind:
            kinds_ok += 1
        else:
            failures.append(
                {
                    "column": col,
                    "reason": "kind_mismatch",
                    "expected": g_kind,
                    "got": pred.kind,
                },
            )

        if canon_gold and (
            canon_gold.lower() in pred.canonical_name.lower()
            or pred.canonical_name.lower() in canon_gold.lower()
        ):
            canon_match += 1

        if not pred.reactome_validated:
            failures.append(
                {
                    "column": col,
                    "reason": "unvalidated",
                    "expected": True,
                    "got": False,
                },
            )

        for gid in sorted(gold_set - pred_set):
            dual = PHOSPHOINOSITIDE_DUAL.get(col)
            reason: FailureReason = "missing_gold_id"
            if dual is not None and gid in dual:
                if not dual.issubset(pred_set):
                    reason = "missing_chebi_form"
            failures.append(
                {
                    "column": col,
                    "reason": reason,
                    "expected": gid,
                    "got": sorted(pred_set),
                },
            )

        for pid in sorted(pred_set - gold_set):
            failures.append(
                {
                    "column": col,
                    "reason": "extra_predicted_id",
                    "expected": sorted(gold_set),
                    "got": pid,
                },
            )

        if gold_set:
            recalls.append(len(intersect) / len(gold_set))
        else:
            recalls.append(1.0 if not pred_set else 0.0)
        if pred_set:
            precs.append(len(intersect) / len(pred_set))
        else:
            precs.append(1.0 if not gold_set else 0.0)
        uni = gold_set | pred_set
        if uni:
            jaccards.append(len(intersect) / len(uni))
        else:
            jaccards.append(1.0)

        if pred.reactome_validated:
            n_validated += 1

    n = len(columns)
    all_validated = bool(predicted) and all(
        g.reactome_validated for g in predicted.values()
    )

    mean_r = sum(recalls) / len(recalls) if recalls else 0.0
    mean_p = sum(precs) / len(precs) if precs else 0.0
    mean_j = sum(jaccards) / len(jaccards) if jaccards else 0.0

    return {
        "n_columns": n,
        "all_validated": all_validated,
        "n_validated": n_validated,
        "column_recall": mean_r,
        "column_precision": mean_p,
        "mean_jaccard": mean_j,
        "kind_accuracy": (kinds_ok / n) if n else 0.0,
        "canonical_name_match": canon_match,
        "failures": failures,
    }
