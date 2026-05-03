"""Step 5 Phase 4 — full Sachs discovery sweep with incremental, idempotent JSON."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from constraints.constraint_builder import load_priors
from experiments.run_condition import run_condition
from utils.load_data import load_sachs_dataset


def _derive_condition_label(priors_source: str, threshold: float | None) -> str:
    """Match :func:`experiments.run_condition._derive_condition_label` for resumability keys."""
    if priors_source == "none":
        return "C0"
    if priors_source == "omnipath_all":
        return "C0.5"
    if priors_source == "omnipath_reactome_only":
        return "C0.5_reactome_only"
    if priors_source == "oracle":
        return "C5"
    if priors_source == "reactome_llm":
        if threshold == 0.9:
            return "C2"
        if threshold == 0.7:
            return "C3"
        if threshold == 0.6:
            return "C4"
        return f"C_llm_t{threshold}"
    raise ValueError(f"Unknown priors_source for condition label: {priors_source!r}")


DATASET_NAME: Final[str] = "sachs"

PRIORS_PATHS: Final[dict[str, Path]] = {
    "reactome_llm": Path("experiments/causal_priors_sachs.json"),
    "omnipath_all": Path("experiments/floor_priors_sachs_all.json"),
    "omnipath_reactome_only": Path("experiments/floor_priors_sachs_reactome_only.json"),
}

ALGORITHMS: Final[tuple[str, ...]] = ("PC", "GES", "LiNGAM")
SEEDS: Final[tuple[int, ...]] = tuple(range(10))
SWEEP_THRESHOLDS: Final[tuple[float, ...]] = (0.6, 0.7, 0.8, 0.9)


@dataclass(frozen=True)
class _CellSpec:
    priors_source: str
    threshold: float | None
    algorithm: str
    seed: int


def _threshold_key(threshold: float | None) -> str:
    if threshold is None:
        return "none"
    return f"{float(threshold):.2f}"


def _result_sort_key(row: dict[str, Any]) -> tuple[str, str, str, float, int]:
    th = row["threshold"]
    th_sort = -1.0 if th is None else float(th)
    return (
        str(row["condition"]),
        str(row["priors_source"]),
        str(row["algorithm"]),
        th_sort,
        int(row["seed"]),
    )


def _cell_result_key(row: dict[str, Any]) -> tuple[str, str, str, str, int]:
    return (
        str(row["condition"]),
        str(row["priors_source"]),
        str(row["algorithm"]),
        _threshold_key(row["threshold"] if "threshold" in row else None),
        int(row["seed"]),
    )


def _build_cell_matrix() -> list[_CellSpec]:
    cells: list[_CellSpec] = []
    for alg in ALGORITHMS:
        for seed in SEEDS:
            cells.append(_CellSpec("none", None, alg, seed))
    for priors_src in ("omnipath_all", "omnipath_reactome_only", "reactome_llm"):
        for thr in SWEEP_THRESHOLDS:
            for alg in ALGORITHMS:
                for seed in SEEDS:
                    cells.append(_CellSpec(priors_src, thr, alg, seed))
    for alg in ALGORITHMS:
        for seed in SEEDS:
            cells.append(_CellSpec("oracle", None, alg, seed))
    return cells


def _apply_filters(
    cells: list[_CellSpec],
    *,
    only_conditions: list[str] | None,
    only_algorithms: list[str] | None,
    only_seeds: list[int] | None,
) -> list[_CellSpec]:
    """Filter cells; ``only_conditions`` matches :attr:`_CellSpec.priors_source` values."""
    out = cells
    if only_conditions is not None:
        allow_src = frozenset(only_conditions)
        out = [c for c in out if c.priors_source in allow_src]
    if only_algorithms is not None:
        allow_alg = frozenset(only_algorithms)
        out = [c for c in out if c.algorithm in allow_alg]
    if only_seeds is not None:
        allow_seed = frozenset(only_seeds)
        out = [c for c in out if c.seed in allow_seed]
    return out


def row_key_from_spec(cell: _CellSpec) -> tuple[str, str, str, str, int]:
    condition = _derive_condition_label(cell.priors_source, cell.threshold)
    return (
        condition,
        cell.priors_source,
        cell.algorithm,
        _threshold_key(cell.threshold),
        cell.seed,
    )


def _result_sort_key_cell(cell: _CellSpec) -> tuple[str, str, str, float, int]:
    condition = _derive_condition_label(cell.priors_source, cell.threshold)
    th_sort = -1.0 if cell.threshold is None else float(cell.threshold)
    return (condition, cell.priors_source, cell.algorithm, th_sort, cell.seed)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _load_priors_cache() -> dict[str, list[Any] | None]:
    cache: dict[str, list[Any] | None] = {
        "none": None,
        "oracle": None,
    }
    for key, rel in PRIORS_PATHS.items():
        cache[key] = load_priors(rel)
    return cache


def _summarise_payload(results: list[dict[str, Any]]) -> dict[str, Any]:
    n_failed = sum(1 for r in results if r.get("status") == "failed")
    ges_dropped = 0
    for r in results:
        if r.get("algorithm") == "GES":
            dropped = r.get("dropped_due_to_cycle") or []
            ges_dropped += len(dropped)
    return {
        "n_cells_completed": len(results),
        "n_cells_failed": n_failed,
        "ges_total_dropped_due_to_cycle": ges_dropped,
    }


def run_sweep(
    *,
    output_path: Path = Path("experiments/discovery_results_sachs.json"),
    only_conditions: list[str] | None = None,
    only_algorithms: list[str] | None = None,
    only_seeds: list[int] | None = None,
) -> None:
    all_matrix = _build_cell_matrix()
    cells = _apply_filters(
        all_matrix,
        only_conditions=only_conditions,
        only_algorithms=only_algorithms,
        only_seeds=only_seeds,
    )
    cells.sort(key=_result_sort_key_cell)

    n_cells_expected = len(cells)

    results: list[dict[str, Any]] = []
    if output_path.is_file():
        raw = json.loads(output_path.read_text(encoding="utf-8"))
        results = list(raw.get("results") or [])

    existing_keys = {_cell_result_key(r) for r in results}
    pending = [c for c in cells if row_key_from_spec(c) not in existing_keys]

    if pending:
        data_df, true_graph = load_sachs_dataset()
        variable_names = list(data_df.columns)
        data_matrix = data_df.to_numpy()
        priors_cache = _load_priors_cache()

        n_todo = len(pending)
        for i, cell in enumerate(pending, start=1):
            priors = priors_cache.get(cell.priors_source)
            res = run_condition(
                dataset_name=DATASET_NAME,
                data=data_matrix,
                variable_names=variable_names,
                true_graph=true_graph,
                priors=priors,
                priors_source=cell.priors_source,
                algorithm=cell.algorithm,
                threshold=cell.threshold,
                seed=cell.seed,
            )
            results.append(res)
            results.sort(key=_result_sort_key)
            payload = {
                "dataset": DATASET_NAME,
                "n_cells_expected": n_cells_expected,
                "results": results,
                **_summarise_payload(results),
            }
            _atomic_write_json(output_path, payload)

            status = res.get("status", "?")
            shd = res.get("metrics", {}) or {}
            shd_s = shd.get("shd")
            shd_part = f"shd={shd_s}" if shd_s is not None else "shd=n/a"
            thr_disp = _threshold_key(cell.threshold)
            print(
                f"[{i}/{n_todo}] {res['condition']} {cell.algorithm} "
                f"{thr_disp} {cell.seed} -> {status} ({shd_part})"
            )

    n_ok = sum(1 for r in results if r.get("status") == "ok")
    n_fail = sum(1 for r in results if r.get("status") == "failed")
    ges_tot = 0
    for r in results:
        if r.get("algorithm") == "GES":
            ges_tot += len(r.get("dropped_due_to_cycle") or [])

    print(
        f"Summary: cells_in_file={len(results)} expected={n_cells_expected} "
        f"ok={n_ok} failed={n_fail} ges_dropped_edges={ges_tot}"
    )


def main() -> None:
    run_sweep()


if __name__ == "__main__":
    main()
