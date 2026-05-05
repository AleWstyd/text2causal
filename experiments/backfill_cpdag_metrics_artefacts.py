"""Backfill ``cpdag_*`` / ``shd_cpdag`` in experiment JSONs from ``predicted_edges``."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import networkx as nx
import pandas as pd

from evaluation.harness import backfill_cpdag_metrics_in_results

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

Loader = Callable[[], tuple[pd.DataFrame, nx.DiGraph]]

DATASET_LOADERS: dict[str, Loader] = {}


def _register_loaders() -> None:
    from utils.load_data import load_sachs_dataset
    from utils.load_data import (
        load_dream4_psn_dataset,
        load_liverdream_dataset,
    )

    DATASET_LOADERS["sachs"] = load_sachs_dataset
    DATASET_LOADERS["dream4_psn_synthetic"] = load_dream4_psn_dataset
    DATASET_LOADERS["liverdream"] = load_liverdream_dataset


def backfill_json(path: Path) -> int:
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = list(raw.get("results") or [])
    ds = str(raw.get("dataset") or (results[0].get("dataset") if results else ""))
    loader = DATASET_LOADERS.get(ds)
    if loader is None:
        raise ValueError(f"No loader registered for dataset {ds!r} ({path})")

    data_df, true_graph = loader()
    variable_names = list(data_df.columns)
    n = backfill_cpdag_metrics_in_results(results, variable_names, true_graph)
    raw["results"] = results
    path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return n


def main() -> None:
    _register_loaders()
    for name in (
        "ablation_results_sachs.json",
        "ablation_results_dream4_psn.json",
        "ablation_results_liverdream.json",
    ):
        p = REPO_ROOT / "experiments" / name
        if not p.is_file():
            continue
        n = backfill_json(p)
        print(f"Backfilled CPDAG metrics for {n} ok rows in {p.as_posix()}")


if __name__ == "__main__":
    main()
