"""Step 6 Phase 2 — canonical Sachs ablation sweep (210 algorithmic cells + C-LLM-only)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import networkx as nx

import experiments.run_condition as _rc
from constraints.constraint_builder import load_priors
from evaluation.harness import (
    backfill_directed_metrics_in_results,
    evaluate,
    row_ok_metrics_missing_directed_f1,
)
from utils.load_data import load_sachs_dataset

DATASET_NAME: Final[str] = "sachs"
MATRIX_ID: Final[str] = "step6_canonical"

ORACLE_PRIORS: Final[Path] = Path("experiments/oracle_priors_sachs.json")
FREETEXT_PRIORS: Final[Path] = Path("experiments/freetext_priors_sachs.json")
FLOOR_REACTOME_ONLY: Final[Path] = Path(
    "experiments/floor_priors_sachs_reactome_only.json"
)
CAUSAL_PRIORS: Final[Path] = Path("experiments/causal_priors_sachs.json")
CAUSAL_PRIORS_WITH_FALLBACK: Final[Path] = Path(
    "experiments/causal_priors_sachs_with_fallback.json"
)
PREDICTED_DAG_GML: Final[Path] = Path("experiments/predicted_dag_llm_only_sachs.gml")
PREDICTED_DAG_META: Final[Path] = Path(
    "experiments/predicted_dag_llm_only_sachs.meta.json"
)

_ALGORITHMS: Final[tuple[str, ...]] = ("PC", "GES", "LiNGAM")
_SEEDS: Final[tuple[int, ...]] = tuple(range(10))


@dataclass(frozen=True)
class _AlgorithmicCell:
    condition: str
    priors_source: str
    priors_cache_key: str
    threshold: float | None
    algorithm: str
    seed: int
    lingam_prior_mode: str | None = None


@dataclass(frozen=True)
class _CllmCell:
    condition: str = "C-LLM-only"
    priors_source: str = "reactome_llm"


def _threshold_key(threshold: float | None) -> str:
    if threshold is None:
        return "none"
    return f"{float(threshold):.2f}"


def _algorithm_sort_token(algorithm: str | None) -> str:
    if algorithm is None:
        return "_zzz"
    return str(algorithm)


def result_sort_key(row: dict[str, Any]) -> tuple[str, str, str, float, int]:
    th = row.get("threshold")
    th_sort = -1.0 if th is None else float(th)
    seed = row.get("seed")
    seed_sort = -1 if seed is None else int(seed)
    return (
        str(row["condition"]),
        str(row["priors_source"]),
        _algorithm_sort_token(row.get("algorithm")),
        th_sort,
        seed_sort,
    )


def result_row_key(row: dict[str, Any]) -> tuple[str, str, str, str, int, str]:
    alg = row.get("algorithm")
    alg_part = "_llm_only" if alg is None else str(alg)
    seed = row.get("seed")
    seed_part = -1 if seed is None else int(seed)
    lingam_mode = row.get("lingam_prior_mode")
    lingam_part = "_none" if lingam_mode is None else str(lingam_mode)
    return (
        str(row["condition"]),
        str(row["priors_source"]),
        alg_part,
        _threshold_key(row["threshold"] if "threshold" in row else None),
        seed_part,
        lingam_part,
    )


def _result_row_base_key(row: dict[str, Any]) -> tuple[str, str, str, str, int]:
    """Key without LiNGAM mode, used to retire stale canonical rows."""
    key = result_row_key(row)
    return key[:5]


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _summarise_payload(
    results: list[dict[str, Any]], n_expected: int
) -> dict[str, Any]:
    n_failed = sum(1 for r in results if r.get("status") == "failed")
    ges_dropped = 0
    pc_dropped = 0
    for r in results:
        dropped = r.get("dropped_due_to_cycle") or []
        if r.get("algorithm") == "GES":
            ges_dropped += len(dropped)
        if r.get("algorithm") == "PC":
            pc_dropped += len(dropped)
    return {
        "dataset": DATASET_NAME,
        "matrix": MATRIX_ID,
        "n_cells_expected": n_expected,
        "n_cells_completed": len(results),
        "n_cells_failed": n_failed,
        "ges_total_dropped_due_to_cycle": ges_dropped,
        "pc_total_dropped_due_to_cycle": pc_dropped,
    }


def _predicted_edges_list(graph: nx.DiGraph) -> list[list[str]]:
    edges = [(str(u), str(v)) for u, v in graph.edges()]
    edges.sort(key=lambda t: (t[0], t[1]))
    return [list(pair) for pair in edges]


def _json_safe_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in metrics.items()}


def _build_sachs_algorithmic_matrix() -> list[_AlgorithmicCell]:
    cells: list[_AlgorithmicCell] = []
    for alg in _ALGORITHMS:
        for seed in _SEEDS:
            cells.append(
                _AlgorithmicCell(
                    condition="C0",
                    priors_source="none",
                    priors_cache_key="none",
                    threshold=None,
                    algorithm=alg,
                    seed=seed,
                )
            )
    for alg in _ALGORITHMS:
        for seed in _SEEDS:
            cells.append(
                _AlgorithmicCell(
                    condition="C0.5",
                    priors_source="omnipath_reactome_only",
                    priors_cache_key="omnipath_reactome_only",
                    threshold=0.7,
                    algorithm=alg,
                    seed=seed,
                    lingam_prior_mode=("forbidden_only" if alg == "LiNGAM" else None),
                )
            )
    for alg in _ALGORITHMS:
        for seed in _SEEDS:
            cells.append(
                _AlgorithmicCell(
                    condition="C1",
                    priors_source="freetext_llm",
                    priors_cache_key="freetext_llm",
                    threshold=0.7,
                    algorithm=alg,
                    seed=seed,
                    lingam_prior_mode=("forbidden_only" if alg == "LiNGAM" else None),
                )
            )
    for thr, cond in ((0.9, "C2"), (0.7, "C3"), (0.6, "C4")):
        for alg in _ALGORITHMS:
            for seed in _SEEDS:
                cells.append(
                    _AlgorithmicCell(
                        condition=cond,
                        priors_source="reactome_llm",
                        priors_cache_key="reactome_llm",
                        threshold=thr,
                        algorithm=alg,
                        seed=seed,
                        lingam_prior_mode=(
                            "forbidden_only" if alg == "LiNGAM" else None
                        ),
                    )
                )
    for alg in _ALGORITHMS:
        for seed in _SEEDS:
            cells.append(
                _AlgorithmicCell(
                    condition="C3+ft",
                    priors_source="reactome_llm_with_freetext_fallback",
                    priors_cache_key="reactome_llm_with_freetext_fallback",
                    threshold=0.7,
                    algorithm=alg,
                    seed=seed,
                    lingam_prior_mode=("forbidden_only" if alg == "LiNGAM" else None),
                )
            )
    for alg in _ALGORITHMS:
        for seed in _SEEDS:
            cells.append(
                _AlgorithmicCell(
                    condition="C5",
                    priors_source="oracle",
                    priors_cache_key="oracle",
                    threshold=None,
                    algorithm=alg,
                    seed=seed,
                    lingam_prior_mode=("forbidden_only" if alg == "LiNGAM" else None),
                )
            )
    cells.sort(
        key=lambda c: (
            c.condition,
            c.priors_source,
            c.algorithm,
            -1.0 if c.threshold is None else float(c.threshold),
            c.seed,
        )
    )
    return cells


def _apply_cell_filters(
    algo_cells: list[_AlgorithmicCell],
    cllm: _CllmCell,
    *,
    only_conditions: list[str] | None,
    only_algorithms: list[str] | None,
    only_seeds: list[int] | None,
) -> tuple[list[_AlgorithmicCell], _CllmCell | None]:
    out_algo = list(algo_cells)
    out_cllm: _CllmCell | None = cllm

    if only_conditions is not None:
        allow = frozenset(only_conditions)
        out_algo = [c for c in out_algo if c.condition in allow]
        if cllm.condition not in allow:
            out_cllm = None

    if only_algorithms is not None:
        allow_alg = frozenset(only_algorithms)
        out_algo = [c for c in out_algo if c.algorithm in allow_alg]
        out_cllm = None

    if only_seeds is not None:
        allow_s = frozenset(only_seeds)
        out_algo = [c for c in out_algo if c.seed in allow_s]
        out_cllm = None

    return out_algo, out_cllm


def _load_priors_cache() -> dict[str, list[Any] | None]:
    return {
        "none": None,
        "omnipath_reactome_only": load_priors(FLOOR_REACTOME_ONLY),
        "freetext_llm": load_priors(FREETEXT_PRIORS),
        "reactome_llm": load_priors(CAUSAL_PRIORS),
        "reactome_llm_with_freetext_fallback": load_priors(CAUSAL_PRIORS_WITH_FALLBACK),
        "oracle": load_priors(ORACLE_PRIORS),
    }


def _run_cllm_cell(
    *,
    true_graph: nx.DiGraph,
    gml_path: Path,
    meta_path: Path,
) -> dict[str, Any]:
    predicted = nx.read_gml(gml_path, label="label")

    def _aligned_names() -> bool:
        pred_ns = {str(n) for n in predicted.nodes()}
        true_ns = {str(n) for n in true_graph.nodes()}
        return pred_ns == true_ns and all(n in predicted for n in true_graph.nodes())

    if not _aligned_names():
        mapping: dict[Any, str] = {}
        for n in list(predicted.nodes()):
            data = predicted.nodes[n]
            lbl = data.get("label")
            if lbl is not None:
                mapping[n] = str(lbl)
        if mapping:
            predicted = nx.relabel_nodes(predicted, mapping)

    metrics = evaluate(predicted, true_graph)
    gml_str = gml_path.as_posix()
    meta_str = meta_path.as_posix()
    meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta_obj, dict):
        raise TypeError(f"Meta JSON must be an object: {meta_path}")
    meta_thr = meta_obj.get("threshold")
    priors_hash = meta_obj.get("source_priors_hash")
    return {
        "dataset": DATASET_NAME,
        "condition": "C-LLM-only",
        "algorithm": None,
        "seed": None,
        "threshold": None,
        "priors_source": "reactome_llm",
        "status": "ok",
        "error": None,
        "metrics": _json_safe_metrics(metrics),
        "predicted_edges": _predicted_edges_list(predicted),
        "constraint_summary": None,
        "dropped_due_to_cycle": [],
        "notes": {
            "meta_path": meta_str,
            "predicted_dag_path": gml_str,
            "n_nodes": int(predicted.number_of_nodes()),
            "n_edges": int(predicted.number_of_edges()),
            "threshold": meta_thr,
            "source_priors_hash": priors_hash,
        },
    }


def run_ablation(
    *,
    dataset_name: str = "sachs",
    output_path: Path = Path("experiments/ablation_results_sachs.json"),
    only_conditions: list[str] | None = None,
    only_algorithms: list[str] | None = None,
    only_seeds: list[int] | None = None,
) -> None:
    if dataset_name != "sachs":
        raise NotImplementedError(f"runner does not yet support dataset {dataset_name}")

    full_algo = _build_sachs_algorithmic_matrix()
    cllm_cell = _CllmCell()
    algo_scheduled, cllm_scheduled = _apply_cell_filters(
        full_algo,
        cllm_cell,
        only_conditions=only_conditions,
        only_algorithms=only_algorithms,
        only_seeds=only_seeds,
    )
    n_expected = len(algo_scheduled) + (1 if cllm_scheduled is not None else 0)

    results: list[dict[str, Any]] = []
    if output_path.is_file():
        raw = json.loads(output_path.read_text(encoding="utf-8"))
        results = list(raw.get("results") or [])

    def algo_key(c: _AlgorithmicCell) -> tuple[str, str, str, str, int, str]:
        return (
            c.condition,
            c.priors_source,
            c.algorithm,
            _threshold_key(c.threshold),
            c.seed,
            "_none" if c.lingam_prior_mode is None else c.lingam_prior_mode,
        )

    expected_algo_keys = {algo_key(c) for c in algo_scheduled}
    expected_keys = set(expected_algo_keys)
    if cllm_scheduled is not None:
        expected_keys.add(
            ("C-LLM-only", "reactome_llm", "_llm_only", "none", -1, "_none")
        )
    expected_bases = {key[:5] for key in expected_keys}
    # Step 6.7 changes the canonical Reactome+LLM LiNGAM rows from dense
    # matrices to forbidden-only sparse matrices. Remove stale rows with the
    # same condition/source/algorithm/threshold/seed but the old mode so the
    # result file remains a true canonical matrix rather than an append-only log.
    results = [
        row
        for row in results
        if _result_row_base_key(row) not in expected_bases
        or result_row_key(row) in expected_keys
    ]
    existing = {result_row_key(r) for r in results}

    pending_algo = [c for c in algo_scheduled if algo_key(c) not in existing]
    pending_cllm = False
    if cllm_scheduled is not None:
        cllm_k = ("C-LLM-only", "reactome_llm", "_llm_only", "none", -1, "_none")
        if cllm_k not in existing:
            pending_cllm = True

    def _print_summary(results_: list[dict[str, Any]]) -> None:
        n_ok = sum(1 for r in results_ if r.get("status") == "ok")
        n_fail = sum(1 for r in results_ if r.get("status") == "failed")
        ges_tot = sum(
            len(r.get("dropped_due_to_cycle") or [])
            for r in results_
            if r.get("algorithm") == "GES"
        )
        pc_tot = sum(
            len(r.get("dropped_due_to_cycle") or [])
            for r in results_
            if r.get("algorithm") == "PC"
        )
        print(
            f"Summary: cells_in_file={len(results_)} expected={n_expected} "
            f"ok={n_ok} failed={n_fail} ges_dropped_edges={ges_tot} "
            f"pc_dropped_edges={pc_tot}"
        )

    needs_metric_backfill = any(row_ok_metrics_missing_directed_f1(r) for r in results)

    if not pending_algo and not pending_cllm:
        if needs_metric_backfill:
            data_df, true_graph = load_sachs_dataset()
            variable_names = list(data_df.columns)
            n_bf = backfill_directed_metrics_in_results(
                results, variable_names, true_graph
            )
            payload = {
                "results": results,
                **_summarise_payload(results, n_expected),
            }
            _atomic_write_json(output_path, payload)
            print(f"Backfilled directed metrics for {n_bf} ok rows.")
        _print_summary(results)
        return

    data_df, true_graph = load_sachs_dataset()
    variable_names = list(data_df.columns)
    if needs_metric_backfill:
        n_bf = backfill_directed_metrics_in_results(results, variable_names, true_graph)
        payload = {
            "results": results,
            **_summarise_payload(results, n_expected),
        }
        _atomic_write_json(output_path, payload)
        print(f"Backfilled directed metrics for {n_bf} ok rows.")

    data_matrix = data_df.to_numpy()
    priors_cache = _load_priors_cache()

    n_todo = len(pending_algo) + (1 if pending_cllm else 0)
    done = 0

    for cell in pending_algo:
        done += 1
        priors = priors_cache[cell.priors_cache_key]
        res = _rc.run_condition(
            dataset_name=DATASET_NAME,
            data=data_matrix,
            variable_names=variable_names,
            true_graph=true_graph,
            priors=priors,
            priors_source=cell.priors_source,
            algorithm=cell.algorithm,
            threshold=cell.threshold,
            seed=cell.seed,
            lingam_prior_mode=cell.lingam_prior_mode,
        )
        if cell.condition == "C0.5":
            res["condition"] = "C0.5"
        elif res["condition"] != cell.condition:
            res["condition"] = cell.condition

        results.append(res)
        results.sort(key=result_sort_key)
        payload = {
            "results": results,
            **_summarise_payload(results, n_expected),
        }
        _atomic_write_json(output_path, payload)

        status = res.get("status", "?")
        mt = res.get("metrics") or {}
        shd_s = mt.get("shd")
        shd_part = f"shd={shd_s}" if shd_s is not None else "shd=n/a"
        print(
            f"[{done}/{n_todo}] {res['condition']} {cell.algorithm} "
            f"{_threshold_key(cell.threshold)} {cell.seed} -> {status} ({shd_part})"
        )

    if pending_cllm:
        done += 1
        res = _run_cllm_cell(
            true_graph=true_graph,
            gml_path=PREDICTED_DAG_GML,
            meta_path=PREDICTED_DAG_META,
        )
        results.append(res)
        results.sort(key=result_sort_key)
        payload = {
            "results": results,
            **_summarise_payload(results, n_expected),
        }
        _atomic_write_json(output_path, payload)
        mt = res.get("metrics") or {}
        print(
            f"[{done}/{n_todo}] C-LLM-only _llm_only none -1 -> ok "
            f"(shd={mt.get('shd')})"
        )

    _print_summary(results)


def main() -> None:
    run_ablation()


if __name__ == "__main__":
    main()
