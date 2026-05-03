#!/usr/bin/env python3
"""Run batched Sachs variable grounding + evaluation; write artefacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from grounding.evaluate import evaluate_grounding
from grounding.ground import ground_columns
from utils.load_data import load_sachs_dataset

GOLD_PATH = Path("experiments/grounding_sachs_gold.json")
OUT_PATH = Path("experiments/grounding_sachs.json")
CACHE_LL = Path("cache/llm")

SACHS_DATASET_DESCRIPTION = (
    "Sachs et al. (2005) protein-signalling data: 7,466 flow-cytometry instances × "
    "11 columns measuring phosphorylated proteins and phospholipids in human primary "
    "immune cells (CD4+ T-cells and B-cells), including observational and "
    "interventional (stimulation/knockdown) conditions."
)
SACHS_DOMAIN_HINT = (
    "Human cell signalling in primary immune cells (CD4+ T cells, B cells), "
    "measured by intracellular flow cytometry. Pathways covered: MAPK/ERK, "
    "PI3K–Akt, PLCγ, PKC, PKA, JNK, p38, and phosphoinositide lipid metabolism. "
    "\n\nSachs column-name conventions (these resolve common ambiguities; do NOT "
    "guess UniProt IDs you don't know — drop anything unfamiliar):\n"
    "- Lowercase 'p' prefix denotes a phosphorylated form (e.g. praf = phospho-RAF1).\n"
    "- 'pmek' is phospho-MEK (MAP2K family — kinases that activate ERK), NOT phospho-ERK / NOT MAPK1/MAPK3.\n"
    "- 'plcg' is phospho-PLCγ (phospholipase C gamma family), NOT a MAP kinase.\n"
    "- 'pakts473' is AKT phosphorylated at Ser473 (the AKT family of PI3K effectors), NOT PAK kinase.\n"
    "- 'p44/42' is the historical molecular-weight name for ERK1/ERK2 (MAPK3/MAPK1).\n"
    "- 'pjnk' is phospho-JNK (the JNK / SAPK family).\n"
    "- 'P38' is the p38 MAPK family (multiple isoforms).\n"
    "- 'PKC' is the pan-PKC family (conventional, novel, atypical isoforms).\n"
    "- 'PKA' is the cAMP-dependent kinase, catalytic subunits.\n"
    "- 'PIP2' and 'PIP3' are phosphoinositide lipids (return BOTH ChEBI forms each, "
    "as required by the system rules)."
)


def main() -> int:
    data, _ = load_sachs_dataset()
    columns = list(data.columns)

    gold_blob = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    gold = gold_blob["groundings"]
    if set(gold) != set(columns):
        missing = set(columns) - set(gold)
        extra = set(gold) - set(columns)
        print(
            "Gold fixture column mismatch with Sachs DataFrame.",
            f"missing_in_gold={sorted(missing)} extra_in_gold={sorted(extra)}",
            file=sys.stderr,
        )
        return 1

    predicted = ground_columns(
        columns,
        SACHS_DATASET_DESCRIPTION,
        SACHS_DOMAIN_HINT,
        cache_dir=CACHE_LL,
    )
    metrics = evaluate_grounding(predicted, gold)

    served = next(iter(predicted.values())).served_model if predicted else None
    cache_files = len(list(CACHE_LL.glob("*.json"))) if CACHE_LL.exists() else 0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                "predicted": {k: v.as_dict() for k, v in predicted.items()},
                "gold": gold,
                "metrics": metrics,
                "served_model": served,
                "cache_files_total_after": cache_files,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"\nWrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
