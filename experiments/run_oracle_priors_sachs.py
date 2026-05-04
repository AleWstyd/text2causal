"""Emit ground-truth oracle priors for Sachs (Step 6 Phase 1 artefact)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from utils.load_data import load_sachs_dataset

OUT_PATH = Path("experiments/oracle_priors_sachs.json")


def _atomic_write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    data_df, true_graph = load_sachs_dataset()
    variable_names = list(data_df.columns)
    n = len(variable_names)
    n_ordered = n * (n - 1)

    pairs: list[dict[str, object]] = []
    for u, v in sorted(true_graph.edges(), key=lambda e: (e[0], e[1])):
        pairs.append(
            {
                "var_a": u,
                "var_b": v,
                "cause": u,
                "effect": v,
                "confidence": 1.0,
                "constraint_type": "hard_required",
                "source": "oracle",
            }
        )

    payload = {
        "n_pairs_total": n_ordered,
        "n_pairs_with_edge": len(pairs),
        "pairs": pairs,
    }
    _atomic_write_json(OUT_PATH, payload)
    print(f"Wrote {OUT_PATH} ({len(pairs)} edges)")


if __name__ == "__main__":
    main()
