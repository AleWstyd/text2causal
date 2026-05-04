"""Generate per-pair LLM priors for Reactome no-context pairs (PR2b)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Final

from reasoning.freetext_fallback import generate_per_pair_freetext_priors
from reasoning.merge_freetext_fallback import infer_column_set_from_priors

REPO_ROOT = Path(__file__).resolve().parents[1]

DATASETS: Final[tuple[tuple[str, Path, str], ...]] = (
    (
        "sachs",
        REPO_ROOT / "experiments" / "causal_priors_sachs.json",
        "Sachs et al. 2005 intracellular signaling dataset; 11 phosphoproteins "
        "and phospholipids measured by multi-parameter flow cytometry in "
        "primary human T cells.",
    ),
    (
        "dream4_psn",
        REPO_ROOT / "experiments" / "causal_priors_dream4_psn.json",
        "DREAM4 Predictive Signaling Network synthetic variant; 7-node human "
        "MAPK / PI3K / JAK-STAT signalling panel.",
    ),
    (
        "liverdream",
        REPO_ROOT / "experiments" / "causal_priors_liverdream.json",
        "DREAM HepG2 hepatocyte phosphoprotein panel; 7 signalling proteins "
        "perturbed with ligands and inhibitors.",
    ),
)

CACHE_LLM = Path("cache/llm")


def _atomic_write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _no_context_as_tuples(raw: object) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.append((str(item[0]), str(item[1])))
    return out


def main() -> None:
    for name, causal_path, description in DATASETS:
        if not causal_path.is_file():
            print(f"[skip {name}] missing {causal_path}")
            continue
        causal = json.loads(causal_path.read_text(encoding="utf-8"))
        if not isinstance(causal, dict):
            raise TypeError(f"Expected object in {causal_path}")
        nc = _no_context_as_tuples(causal.get("no_context_pairs"))
        if not nc:
            print(f"[skip {name}] no no_context_pairs in {causal_path.name}")
            continue

        col = infer_column_set_from_priors(causal)
        vocab = sorted(col)
        payload = generate_per_pair_freetext_priors(
            name,
            description,
            nc,
            vocab,
            cache_dir=CACHE_LLM,
        )
        out_path = REPO_ROOT / "experiments" / f"per_pair_freetext_priors_{name}.json"
        _atomic_write_json(out_path, payload)
        print(
            f"[{name}] per_pair_freetext: n_pairs_queried="
            f"{payload.get('n_pairs_queried')} n_emitted={payload['n_relations_emitted']} "
            f"n_skipped={payload['n_relations_skipped_unknown_var']} "
            f"served_models={payload.get('served_models')} -> {out_path}"
        )


if __name__ == "__main__":
    main()
