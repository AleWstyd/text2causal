#!/usr/bin/env python3
"""Step 4 Phase 3 full sweep: reason over all 110 ordered Sachs pairs.

Reads the locked Step 3 grounding from ``experiments/grounding_sachs.json``
and produces ``experiments/causal_priors_sachs.json`` per the schema in
``docs/step_04_causal_reasoning.md``. Both directions of every unordered
pair are queried so the LLM gets to disagree with itself; conflicts are
logged.

Cache hygiene: every prompt-text or grounding change invalidates every
cached LLM key. With the prompt locked in Phase 2, a second run of this
script must be a 100% cache hit (no LLM calls, identical artefact).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from grounding.ground import Grounding
from reactome.client import ReactomeClient
from reasoning.reason import reason_all_pairs

GROUNDING_PATH = Path("experiments/grounding_sachs.json")
OUT_PATH = Path("experiments/causal_priors_sachs.json")
CACHE_LLM = Path("cache/llm")


def _grounding_from_predicted(
    predicted: dict[str, dict[str, Any]],
) -> dict[str, Grounding]:
    out: dict[str, Grounding] = {}
    for column, entry in predicted.items():
        out[column] = Grounding(
            column=entry.get("column", column),
            kind=entry["kind"],
            ids=list(entry.get("ids") or []),
            canonical_name=entry.get("canonical_name", ""),
            gene_names=list(entry.get("gene_names") or []),
            confidence=float(entry.get("confidence", 0.0)),
            reasoning=entry.get("reasoning", ""),
            reactome_validated=bool(entry.get("reactome_validated", False)),
            served_model=entry.get("served_model"),
        )
    return out


def main() -> int:
    blob = json.loads(GROUNDING_PATH.read_text(encoding="utf-8"))
    grounding = _grounding_from_predicted(blob["predicted"])

    client = ReactomeClient()
    started = time.time()
    payload = reason_all_pairs(
        grounding=grounding,
        reactome_client=client,
        cache_dir=CACHE_LLM,
    )
    payload["wall_clock_seconds"] = round(time.time() - started, 2)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    summary = {
        "n_pairs_total": payload["n_pairs_total"],
        "n_with_context": payload["n_with_context"],
        "n_no_context": payload["n_no_context"],
        "n_high_confidence": payload["n_high_confidence"],
        "n_conflicts": payload["n_conflicts"],
        "served_models": payload["served_models"],
        "wall_clock_seconds": payload["wall_clock_seconds"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nWrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
