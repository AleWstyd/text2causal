#!/usr/bin/env python3
"""Build the OmniPath C0.5 floor priors for Sachs (Step 4 Phase 1).

Emits two artefacts side-by-side under ``experiments/``:

- ``floor_priors_sachs_all.json`` — every directed OmniPath source counts
  (``source_filter='all'``); the "any structured ontology" comparison.
- ``floor_priors_sachs_reactome_only.json`` — restricts ``sources`` to the
  Reactome-named tokens; the apples-to-apples comparison with the LLM path
  which itself draws only from Reactome.

Reuses the grounding committed at ``experiments/grounding_sachs.json``;
this script does NOT regenerate Step 3.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from grounding.ground import Grounding
from omnipath_floor.floor import build_floor_priors, write_floor_priors

GROUNDING_PATH = Path("experiments/grounding_sachs.json")
OUT_ALL = Path("experiments/floor_priors_sachs_all.json")
OUT_REACTOME = Path("experiments/floor_priors_sachs_reactome_only.json")


def _grounding_from_predicted(
    predicted: dict[str, dict[str, Any]],
) -> dict[str, Grounding]:
    """Re-hydrate ``Grounding`` records from the persisted Step 3 artefact."""

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

    payload_all = build_floor_priors(grounding, source_filter="all")
    payload_reactome = build_floor_priors(grounding, source_filter="reactome_only")

    write_floor_priors(OUT_ALL, payload_all)
    write_floor_priors(OUT_REACTOME, payload_reactome)

    summary = {
        "all": {
            "n_pairs_with_edge": payload_all["n_pairs_with_edge"],
            "n_hard_required": payload_all["n_hard_required"],
            "n_soft_prior": payload_all["n_soft_prior"],
            "n_unknown_kept": payload_all["n_unknown_kept"],
            "n_pairs_skipped_metabolite": payload_all["n_pairs_skipped_metabolite"],
        },
        "reactome_only": {
            "n_pairs_with_edge": payload_reactome["n_pairs_with_edge"],
            "n_hard_required": payload_reactome["n_hard_required"],
            "n_soft_prior": payload_reactome["n_soft_prior"],
            "n_unknown_kept": payload_reactome["n_unknown_kept"],
            "n_pairs_skipped_metabolite": payload_reactome[
                "n_pairs_skipped_metabolite"
            ],
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nWrote {OUT_ALL}")
    print(f"Wrote {OUT_REACTOME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
