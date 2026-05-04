"""Multi-LLM robustness check for Reactome-context causal priors."""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path
from typing import Any, Final

import networkx as nx

import experiments.run_condition as _rc
from constraints.constraint_builder import ClaimRecord, load_priors
from experiments.constraint_quality import compute_quality
from grounding.ground import Grounding
from reactome.client import ReactomeClient
from reasoning.reason import default_llm_fetch, reason_all_pairs
from utils.load_data import load_liverdream_dataset, load_sachs_dataset

CACHE_LLM: Final[Path] = Path("cache/llm")
EXPERIMENTS_DIR: Final[Path] = Path("experiments")
SEEDS: Final[tuple[int, ...]] = tuple(range(10))
ALGORITHMS: Final[tuple[str, ...]] = ("PC", "GES", "LiNGAM")
DEFAULT_EXTRA_MODELS: Final[tuple[str, ...]] = ("openai/gpt-oss-120b",)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def _grounding_from_blob(blob: dict[str, Any]) -> dict[str, Grounding]:
    raw = blob.get("predicted") or blob.get("groundings")
    if not isinstance(raw, dict):
        raise ValueError("Grounding blob must contain 'predicted' or 'groundings'")
    out: dict[str, Grounding] = {}
    for column, entry in raw.items():
        out[column] = Grounding(
            column=str(entry.get("column", column)),
            kind=entry["kind"],
            ids=list(entry.get("ids") or []),
            canonical_name=str(entry.get("canonical_name", "")),
            gene_names=list(entry.get("gene_names") or []),
            confidence=float(entry.get("confidence", 0.0)),
            reasoning=str(entry.get("reasoning", "")),
            reactome_validated=bool(entry.get("reactome_validated", False)),
            served_model=entry.get("served_model"),
        )
    return out


def _datasets() -> dict[str, dict[str, Any]]:
    sachs_data, sachs_graph = load_sachs_dataset()
    liver_data, liver_graph = load_liverdream_dataset()
    return {
        "sachs": {
            "data": sachs_data,
            "true_graph": sachs_graph,
            "grounding_path": EXPERIMENTS_DIR / "grounding_sachs.json",
            "canonical_priors_path": EXPERIMENTS_DIR / "causal_priors_sachs.json",
            "output_prefix": "sachs",
        },
        "liverdream": {
            "data": liver_data,
            "true_graph": liver_graph,
            "grounding_path": EXPERIMENTS_DIR / "grounding_liverdream.json",
            "canonical_priors_path": EXPERIMENTS_DIR / "causal_priors_liverdream.json",
            "output_prefix": "liverdream",
        },
    }


def _model_slug(model: str) -> str:
    return (
        model.lower()
        .replace("/", "_")
        .replace(":", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def _run_model_priors(dataset_name: str, cfg: dict[str, Any], model: str) -> Path:
    grounding_blob = json.loads(Path(cfg["grounding_path"]).read_text(encoding="utf-8"))
    grounding = _grounding_from_blob(grounding_blob)
    client = ReactomeClient()

    def _fetch(messages: list[dict[str, Any]], cache_dir: Path) -> dict[str, Any]:
        return default_llm_fetch(messages, cache_dir=cache_dir, model=model)

    payload = reason_all_pairs(
        grounding=grounding,
        reactome_client=client,
        llm_fetch=_fetch,
        cache_dir=CACHE_LLM,
    )
    payload["dataset"] = dataset_name
    payload["robustness_model"] = model
    out_path = (
        EXPERIMENTS_DIR / f"multi_llm_priors_{dataset_name}_{_model_slug(model)}.json"
    )
    _atomic_write_json(out_path, payload)
    return out_path


def _forward_edges(
    priors: list[ClaimRecord], threshold: float = 0.7
) -> set[tuple[str, str]]:
    return {
        (claim.cause, claim.effect)
        for claim in priors
        if claim.confidence >= threshold
        and claim.constraint_type in {"hard_required", "soft_prior"}
        and claim.cause != "unknown"
        and claim.effect != "unknown"
    }


def _jaccard(a: set[tuple[str, str]], b: set[tuple[str, str]]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _quality(path: Path, true_graph: nx.DiGraph) -> dict[str, Any]:
    true_edges = {tuple(edge) for edge in true_graph.edges()}
    n_pairs = true_graph.number_of_nodes() * (true_graph.number_of_nodes() - 1)
    return compute_quality(
        load_priors(path),
        true_edges,
        confidence_threshold=0.7,
        n_total_pairs=n_pairs,
    )


def _c3_f1(path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    data = cfg["data"].to_numpy()
    variables = list(cfg["data"].columns)
    true_graph = cfg["true_graph"]
    priors = load_priors(path)
    out: dict[str, Any] = {}
    for algorithm in ALGORITHMS:
        vals: list[float] = []
        statuses: list[str] = []
        for seed in SEEDS:
            row = _rc.run_condition(
                dataset_name=str(cfg["output_prefix"]),
                data=data,
                variable_names=variables,
                true_graph=true_graph,
                priors=priors,
                priors_source="reactome_llm",
                algorithm=algorithm,
                threshold=0.7,
                seed=seed,
                lingam_prior_mode=("forbidden_only" if algorithm == "LiNGAM" else None),
            )
            statuses.append(str(row["status"]))
            metrics = row.get("metrics") or {}
            if row.get("status") == "ok" and metrics.get("f1") is not None:
                vals.append(float(metrics["f1"]))
        out[algorithm] = {
            "mean_f1": statistics.mean(vals) if vals else None,
            "std_f1": statistics.stdev(vals) if len(vals) >= 2 else 0.0,
            "n_ok": len(vals),
            "n_failed": statuses.count("failed"),
        }
    return out


def _models_from_env() -> tuple[str, ...]:
    raw = os.environ.get("MULTI_LLM_MODELS", "").strip()
    if raw:
        return tuple(model.strip() for model in raw.split(",") if model.strip())
    return DEFAULT_EXTRA_MODELS


def main() -> None:
    datasets = _datasets()
    models = _models_from_env()
    output_priors: dict[str, dict[str, str]] = {}
    summary: dict[str, Any] = {
        "models": list(models),
        "datasets": {},
    }

    for dataset_name, cfg in datasets.items():
        canonical_path = Path(cfg["canonical_priors_path"])
        canonical_priors = load_priors(canonical_path)
        canonical_edges = _forward_edges(canonical_priors)
        dataset_out: dict[str, Any] = {
            "canonical_path": canonical_path.as_posix(),
            "canonical_forward_edges_at_0_7": [
                list(edge) for edge in sorted(canonical_edges)
            ],
            "models": {},
        }
        output_priors[dataset_name] = {}
        for model in models:
            priors_path = _run_model_priors(dataset_name, cfg, model)
            output_priors[dataset_name][model] = priors_path.as_posix()
            model_priors = load_priors(priors_path)
            model_edges = _forward_edges(model_priors)
            dataset_out["models"][model] = {
                "priors_path": priors_path.as_posix(),
                "forward_edges_at_0_7": [list(edge) for edge in sorted(model_edges)],
                "jaccard_vs_canonical": _jaccard(canonical_edges, model_edges),
                "constraint_quality": _quality(priors_path, cfg["true_graph"]),
                "c3_downstream_f1": _c3_f1(priors_path, cfg),
            }
        summary["datasets"][dataset_name] = dataset_out

    _atomic_write_json(EXPERIMENTS_DIR / "multi_llm_agreement.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
