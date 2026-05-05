"""One-shot: duplicate each ``gold_version=original`` ablation row for ``mooij2020`` scoring."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from evaluation.harness import (
    digraph_from_predicted_edges_row,
    evaluate,
)
from experiments import runner as r
from utils.load_data import load_sachs_dataset


def main() -> None:
    output_path = Path("experiments/ablation_results_sachs.json")
    raw: dict[str, Any] = json.loads(output_path.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = list(raw.get("results") or [])

    data_df, _g0 = load_sachs_dataset("original")
    variable_names = list(data_df.columns)
    _, true_mooij = load_sachs_dataset("mooij2020")

    for row in results:
        if "gold_version" not in row:
            row["gold_version"] = "original"

    existing = {r.result_row_key(row) for row in results}
    additions: list[dict[str, Any]] = []

    for row in results:
        if str(row.get("gold_version")) != "original":
            continue
        base = r.result_row_key(row)
        mkey = base[:-1] + ("mooij2020",)
        if mkey in existing:
            continue
        new_row = json.loads(json.dumps(row))
        new_row["gold_version"] = "mooij2020"
        if new_row.get("status") == "ok":
            pred = digraph_from_predicted_edges_row(new_row, variable_names)
            fresh = evaluate(pred, true_mooij)
            m = new_row.setdefault("metrics", {})
            for k, v in fresh.items():
                m[k] = float(v)
        additions.append(new_row)
        existing.add(mkey)

    if not additions:
        print("No new mooij2020 rows added (already complete).")
        return

    results.extend(additions)
    results.sort(key=r.result_sort_key)

    full_algo = r._expand_algo_with_gold(r._build_sachs_algorithmic_matrix())
    cllm_cells = r._build_cllm_cells()
    n_expected = len(full_algo) + len(cllm_cells)

    raw["results"] = results
    raw.update(r._summarise_payload(results, n_expected))
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, output_path)
    print(
        f"Added {len(additions)} mooij2020 rows; total={len(results)} expected={n_expected}"
    )


if __name__ == "__main__":
    main()
