"""Step 6 Phase 1 — C1 free-text LLM priors for Sachs via cached chat_completion."""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any, cast

from llm import extract_relations
from llm.client import chat_completion
from llm.prompts import RELATION_PROMPT
from utils.load_data import load_sachs_dataset

BACKGROUND_PATH = Path("data/sachs/background.txt")
OUT_PATH = Path("experiments/freetext_priors_sachs.json")
CACHE_LLM = Path("cache/llm")

_LOG = logging.getLogger(__name__)


def _atomic_write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _constraint_from_relation(relation_type: str, confidence: float) -> str:
    """Map legacy relation_type + confidence to Step-4 constraint_type."""
    rt = relation_type.strip().lower() if relation_type else ""
    if rt == "required":
        if confidence >= 0.9:
            return "hard_required"
        return "soft_prior"
    if rt == "forbidden":
        return "hard_forbidden_reverse"
    return "unknown"


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    data_df, _true_graph = load_sachs_dataset()
    columns = list(data_df.columns)
    col_set = set(columns)
    background_text = BACKGROUND_PATH.read_text(encoding="utf-8")
    prompt = RELATION_PROMPT.format(variables=", ".join(columns), text=background_text)

    response = cast(
        dict[str, Any],
        chat_completion(
            messages=[{"role": "user", "content": prompt}], cache_dir=CACHE_LLM
        ),
    )
    served_id = str(response.get("model") or "")
    choices = response.get("choices") or []
    content = ""
    if choices and isinstance(choices[0], dict):
        msg = choices[0].get("message") or {}
        if isinstance(msg, dict):
            content = str(msg.get("content") or "").strip()

    to_parse = extract_relations._unwrap_json_text(content)
    try:
        relations = json.loads(to_parse)
    except json.JSONDecodeError as exc:
        _LOG.error("JSON parse failed: %s", exc)
        relations = []

    if not isinstance(relations, list):
        _LOG.error("Expected a JSON array of relations, got %s", type(relations))
        relations = []

    served_models = [served_id] if served_id else []

    skipped = 0
    by_key: dict[tuple[str, str], dict[str, Any]] = {}

    for rel in relations:
        if not isinstance(rel, dict):
            continue
        try:
            cause = str(rel["cause"])
            effect = str(rel["effect"])
            confidence = float(rel["confidence"])
            relation_type = str(rel.get("relation_type", ""))
        except (KeyError, TypeError, ValueError) as exc:
            _LOG.info("Skipping malformed relation %r: %s", rel, exc)
            continue

        if cause not in col_set or effect not in col_set:
            skipped += 1
            _LOG.info(
                "Skipping relation with unknown variable(s): %s -> %s",
                cause,
                effect,
            )
            continue
        if cause == effect:
            skipped += 1
            _LOG.info("Skipping self-loop: %s -> %s", cause, effect)
            continue

        ctype = _constraint_from_relation(relation_type, confidence)
        rec: dict[str, Any] = {
            "var_a": cause,
            "var_b": effect,
            "cause": cause,
            "effect": effect,
            "confidence": confidence,
            "constraint_type": ctype,
            "source": "freetext_llm",
            "served_models": list(served_models),
        }
        by_key[(cause, effect)] = rec

    pairs_sorted = sorted(
        by_key.values(), key=lambda r: (str(r["var_a"]), str(r["var_b"]))
    )
    type_counts = Counter(str(p["constraint_type"]) for p in pairs_sorted)

    payload: dict[str, Any] = {
        "background_text_path": "data/sachs/background.txt",
        "n_hard_forbidden_reverse": type_counts.get("hard_forbidden_reverse", 0),
        "n_hard_required": type_counts.get("hard_required", 0),
        "n_relations_emitted": len(pairs_sorted),
        "n_relations_skipped_unknown_var": skipped,
        "n_soft_prior": type_counts.get("soft_prior", 0),
        "n_unknown": type_counts.get("unknown", 0),
        "pairs": pairs_sorted,
        "served_models": list(served_models),
    }
    _atomic_write_json(OUT_PATH, payload)

    print(json.dumps({k: v for k, v in payload.items() if k != "pairs"}, indent=2))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
