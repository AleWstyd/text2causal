"""Merge free-text LLM priors into Reactome+LLM priors for the no-context stratum."""

from __future__ import annotations

import copy
from typing import Any


def _unordered(var_a: str, var_b: str) -> tuple[str, str]:
    return tuple(sorted((var_a, var_b)))


def infer_column_set_from_priors(reactome_priors: dict[str, Any]) -> set[str]:
    """Variables to consider for fallback matching (vocabulary field or inferred from pairs)."""
    vocab = reactome_priors.get("vocabulary")
    if isinstance(vocab, list) and vocab:
        return {str(x) for x in vocab}
    out: set[str] = set()
    for item in reactome_priors.get("pairs") or []:
        if isinstance(item, dict):
            out.add(str(item["var_a"]))
            out.add(str(item["var_b"]))
    for item in reactome_priors.get("no_context_pairs") or []:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.add(str(item[0]))
            out.add(str(item[1]))
    return out


def _freetext_index(
    freetext_priors: dict[str, Any], column_set: set[str]
) -> dict[tuple[str, str], dict[str, Any]]:
    raw = freetext_priors.get("pairs") or []
    if not isinstance(raw, list):
        return {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        va, vb = str(item.get("var_a", "")), str(item.get("var_b", ""))
        if va not in column_set or vb not in column_set:
            continue
        key = _unordered(va, vb)
        if key not in out:
            out[key] = item
    return out


def _count_high_confidence(pairs: list[Any]) -> int:
    n = 0
    for rec in pairs:
        if not isinstance(rec, dict):
            continue
        try:
            conf = float(rec["confidence"])
        except (KeyError, TypeError, ValueError):
            continue
        ct = str(rec.get("constraint_type", ""))
        if conf >= 0.9 and ct in ("hard_required", "hard_forbidden_reverse"):
            n += 1
    return n


def merge_with_freetext_fallback(
    reactome_priors: dict[str, Any],
    freetext_priors: dict[str, Any],
    *,
    column_set: set[str] | None = None,
) -> dict[str, Any]:
    """Return a new priors blob with ``no_context`` slots filled from free-text claims where available.

    Reactome-grounded rows are unchanged. Each substituted row copies the free-text
    ``cause``, ``effect``, ``confidence``, and ``constraint_type``; ``source`` is
    ``per_pair_freetext_fallback`` when the incoming claim carries that label
    (PR2b), otherwise ``freetext_fallback`` (paragraph extraction).

    If ``column_set`` is ``None``, it is inferred from ``reactome_priors`` via
    :func:`infer_column_set_from_priors`.
    """
    if column_set is None:
        column_set = infer_column_set_from_priors(reactome_priors)
    out = copy.deepcopy(reactome_priors)
    ft_by_pair = _freetext_index(freetext_priors, column_set)

    def _lookup(var_a: str, var_b: str) -> dict[str, Any] | None:
        if var_a not in column_set or var_b not in column_set:
            return None
        return ft_by_pair.get(_unordered(var_a, var_b))

    fallback_applied: list[list[str]] = []

    raw_pairs = out.get("pairs")
    pairs: list[Any] = list(raw_pairs) if isinstance(raw_pairs, list) else []
    new_pairs: list[Any] = []
    for rec in pairs:
        if not isinstance(rec, dict):
            new_pairs.append(rec)
            continue
        if str(rec.get("constraint_type")) == "no_context":
            ft = _lookup(str(rec["var_a"]), str(rec["var_b"]))
            if ft is not None:
                merged = dict(ft)
                merged["var_a"] = str(rec["var_a"])
                merged["var_b"] = str(rec["var_b"])
                merged["source"] = (
                    "per_pair_freetext_fallback"
                    if str(ft.get("source", "")) == "per_pair_freetext_fallback"
                    else "freetext_fallback"
                )
                new_pairs.append(merged)
                fallback_applied.append([merged["var_a"], merged["var_b"]])
            else:
                new_pairs.append(rec)
        else:
            new_pairs.append(rec)

    raw_nc = out.get("no_context_pairs")
    nc_list: list[Any] = raw_nc if isinstance(raw_nc, list) else []
    remaining_nc: list[list[str]] = []
    for item in nc_list:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        va, vb = str(item[0]), str(item[1])
        ft = _lookup(va, vb)
        if ft is not None:
            merged = dict(ft)
            merged["var_a"] = va
            merged["var_b"] = vb
            merged["source"] = (
                "per_pair_freetext_fallback"
                if str(ft.get("source", "")) == "per_pair_freetext_fallback"
                else "freetext_fallback"
            )
            new_pairs.append(merged)
            fallback_applied.append([va, vb])
        else:
            remaining_nc.append([va, vb])

    out["pairs"] = new_pairs
    out["no_context_pairs"] = remaining_nc
    out["n_with_context"] = len(new_pairs)
    out["n_no_context"] = len(remaining_nc)
    out["n_high_confidence"] = _count_high_confidence(new_pairs)
    out["fallback_applied_pairs"] = sorted(fallback_applied, key=lambda p: (p[0], p[1]))
    out["n_fallback_applied"] = len(fallback_applied)
    return out
