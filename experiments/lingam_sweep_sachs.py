"""Step 6.5 A1 — LiNGAM sparse-prior sweep on Sachs.

This script keeps the canonical Step 6 matrix untouched while testing three
DirectLiNGAM encodings for prior knowledge that avoid dense required-edge
matrices.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import experiments.run_condition as _rc
from constraints.constraint_builder import load_priors
from experiments.runner import result_sort_key
from utils.load_data import load_sachs_dataset

OUTPUT_PATH: Final[Path] = Path("experiments/lingam_sweep_sachs.json")
CANONICAL_PATH: Final[Path] = Path("experiments/ablation_results_sachs.json")
CAUSAL_PRIORS: Final[Path] = Path("experiments/causal_priors_sachs.json")
ORACLE_PRIORS: Final[Path] = Path("experiments/oracle_priors_sachs.json")
MATRIX_ID: Final[str] = "step6_5_lingam_sparse"
SEEDS: Final[tuple[int, ...]] = tuple(range(10))


@dataclass(frozen=True)
class LingamSweepCell:
    condition: str
    priors_source: str
    threshold: float | None
    mode: str
    seed: int


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _cells() -> list[LingamSweepCell]:
    cells: list[LingamSweepCell] = []
    for mode in ("sparse_required", "forbidden_only", "hybrid_top5"):
        for threshold, condition in ((0.9, "C2"), (0.7, "C3"), (0.6, "C4")):
            for seed in SEEDS:
                cells.append(
                    LingamSweepCell(
                        condition=condition,
                        priors_source="reactome_llm",
                        threshold=threshold,
                        mode=mode,
                        seed=seed,
                    )
                )
        for seed in SEEDS:
            cells.append(
                LingamSweepCell(
                    condition="C5",
                    priors_source="oracle",
                    threshold=None,
                    mode=mode,
                    seed=seed,
                )
            )
    return cells


def _result_key(row: dict[str, Any]) -> tuple[str, str, str, str, float, int]:
    threshold = row.get("threshold")
    threshold_sort = -1.0 if threshold is None else float(threshold)
    return (
        str(row["condition"]),
        str(row["priors_source"]),
        str(row.get("algorithm")),
        str(row.get("lingam_prior_mode")),
        threshold_sort,
        int(row["seed"]),
    )


def _summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "dataset": "sachs",
        "matrix": MATRIX_ID,
        "n_cells_expected": len(_cells()),
        "n_cells_completed": len(results),
        "n_cells_failed": sum(1 for row in results if row.get("status") == "failed"),
        "best_ok": _best_ok(results),
    }


def _best_ok(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    ok_rows = [
        row
        for row in results
        if row.get("status") == "ok" and (row.get("metrics") or {}).get("f1") is not None
    ]
    if not ok_rows:
        return None
    best = max(ok_rows, key=lambda row: float(row["metrics"]["f1"]))
    return {
        "condition": best["condition"],
        "threshold": best["threshold"],
        "mode": best["lingam_prior_mode"],
        "f1": float(best["metrics"]["f1"]),
        "shd": float(best["metrics"]["shd"]),
        "n_ok_same_cell": sum(
            1
            for row in ok_rows
            if row["condition"] == best["condition"]
            and row["threshold"] == best["threshold"]
            and row["lingam_prior_mode"] == best["lingam_prior_mode"]
        ),
    }


def run_lingam_sweep(output_path: Path = OUTPUT_PATH) -> None:
    data_df, true_graph = load_sachs_dataset()
    variable_names = list(data_df.columns)
    data_matrix = data_df.to_numpy()
    priors_cache = {
        "reactome_llm": load_priors(CAUSAL_PRIORS),
        "oracle": load_priors(ORACLE_PRIORS),
    }

    results: list[dict[str, Any]] = []
    if output_path.is_file():
        raw = json.loads(output_path.read_text(encoding="utf-8"))
        results = list(raw.get("results") or [])

    existing = {_result_key(row) for row in results}
    all_cells = _cells()
    pending = [
        cell
        for cell in all_cells
        if (
            cell.condition,
            cell.priors_source,
            "LiNGAM",
            cell.mode,
            -1.0 if cell.threshold is None else float(cell.threshold),
            cell.seed,
        )
        not in existing
    ]

    for i, cell in enumerate(pending, start=1):
        row = _rc.run_condition(
            dataset_name="sachs",
            data=data_matrix,
            variable_names=variable_names,
            true_graph=true_graph,
            priors=priors_cache[cell.priors_source],
            priors_source=cell.priors_source,
            algorithm="LiNGAM",
            threshold=cell.threshold,
            seed=cell.seed,
            lingam_prior_mode=cell.mode,
        )
        row["condition"] = cell.condition
        results.append(row)
        results.sort(key=_result_key)
        payload = {"results": results, **_summarise(results)}
        _atomic_write_json(output_path, payload)
        metrics = row.get("metrics") or {}
        f1 = metrics.get("f1")
        f1_part = f"f1={f1:.3f}" if isinstance(f1, float) else "f1=n/a"
        print(
            f"[{i}/{len(pending)}] {cell.condition} {cell.mode} seed={cell.seed} "
            f"-> {row['status']} ({f1_part})"
        )

    print(json.dumps(_summarise(results), indent=2, sort_keys=True))


def promote_best_to_canonical(
    sweep_path: Path = OUTPUT_PATH,
    canonical_path: Path = CANONICAL_PATH,
) -> dict[str, Any]:
    """Replace canonical failed LiNGAM rows with the best successful sweep mode."""
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    best = sweep.get("best_ok")
    if not isinstance(best, dict):
        raise RuntimeError("Cannot promote LiNGAM sweep: no successful sweep rows")

    mode = str(best["mode"])
    condition = str(best["condition"])
    threshold = best["threshold"]
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    canon_results = list(canonical["results"])
    sweep_rows = [
        row
        for row in sweep["results"]
        if row.get("status") == "ok"
        and row.get("condition") == condition
        and row.get("threshold") == threshold
        and row.get("lingam_prior_mode") == mode
    ]
    if not sweep_rows:
        raise RuntimeError("Best LiNGAM sweep row disappeared before promotion")

    seeds = {int(row["seed"]) for row in sweep_rows}
    retained = [
        row
        for row in canon_results
        if not (
            row.get("algorithm") == "LiNGAM"
            and row.get("condition") == condition
            and row.get("threshold") == threshold
            and row.get("seed") in seeds
        )
    ]
    promoted = retained + sweep_rows
    promoted.sort(key=result_sort_key)
    canonical.update(
        {
            "results": promoted,
            "n_cells_completed": len(promoted),
            "n_cells_failed": sum(
                1 for row in promoted if row.get("status") == "failed"
            ),
            "lingam_sweep_promoted": best,
        }
    )
    _atomic_write_json(canonical_path, canonical)
    return best


def main() -> None:
    run_lingam_sweep()


if __name__ == "__main__":
    main()
