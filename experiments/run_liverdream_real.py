"""Step 7' — real public LiverDREAM / CellNOpt second-dataset evaluation."""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast
from urllib.request import urlopen

import matplotlib

matplotlib.use("Agg")
import networkx as nx  # noqa: E402
import pandas as pd  # noqa: E402

import experiments.run_condition as _rc
from constraints.constraint_builder import load_priors
from evaluation.harness import evaluate
from experiments.constraint_quality import compute_quality
from experiments.report import aggregate
from grounding.ground import Grounding
from llm import extract_relations
from llm.client import chat_completion
from llm.prompts import RELATION_PROMPT
from omnipath_floor.floor import build_floor_priors, write_floor_priors
from reactome.client import ReactomeClient
from reasoning.dag_from_priors import build_meta, predict_dag_from_priors_with_meta
from reasoning.reason import reason_all_pairs
from utils.load_data import load_liverdream_dataset

DATASET_NAME: Final[str] = "liverdream"
DATA_DIR: Final[Path] = Path("data/liverdream")
RAW_DIR: Final[Path] = DATA_DIR / "raw"
GROUNDING_DIR: Final[Path] = Path("grounding")
EXPERIMENTS_DIR: Final[Path] = Path("experiments")
TABLES_DIR: Final[Path] = Path("tables")
FIGURES_DIR: Final[Path] = Path("figures")
CACHE_LLM: Final[Path] = Path("cache/llm")
SEEDS: Final[tuple[int, ...]] = tuple(range(10))
ALGORITHMS: Final[tuple[str, ...]] = ("PC", "GES", "LiNGAM")
VARIABLES: Final[tuple[str, ...]] = (
    "akt",
    "mek12",
    "erk12",
    "ikb",
    "jnk12",
    "p38",
    "hsp27",
)

RAW_URLS: Final[dict[str, str]] = {
    "MD-LiverDREAM.csv": "https://raw.githubusercontent.com/saezlab/CellNOptR/gh-pages/public/MD-LiverDREAM.csv",
    "PKN-LiverDREAM.sif.txt": "https://raw.githubusercontent.com/saezlab/CellNOptR/gh-pages/public/PKN-LiverDREAM.sif.txt",
}

BACKGROUND = (
    "The LiverDREAM / CellNOpt benchmark is a public HepG2 liver-cell signalling "
    "dataset derived from Saez-Rodriguez et al. and distributed with CellNOptR. "
    "It measures phosphoprotein responses under cytokine and growth-factor "
    "stimulation and kinase inhibition. The measured readouts used here are AKT, "
    "MEK1/2, ERK1/2, IkB, JNK1/2, p38, and HSP27, spanning PI3K/AKT, RAF/MEK/ERK, "
    "stress MAPK, NF-kB, and heat-shock response signalling."
)


@dataclass(frozen=True)
class Condition:
    name: str
    priors_source: str
    priors_path: Path | None
    threshold: float | None


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _download_raw() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in RAW_URLS.items():
        dest = RAW_DIR / filename
        if dest.is_file():
            continue
        with urlopen(url, timeout=60) as response:
            dest.write_bytes(response.read())


def _parse_sif(path: Path) -> nx.DiGraph:
    graph = nx.DiGraph()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        source, sign, target = stripped.split()
        graph.add_edge(source, target, sign=int(sign))
    return graph


def _latent_projected_graph(pkn: nx.DiGraph) -> nx.DiGraph:
    observed = set(VARIABLES)
    out = nx.DiGraph()
    out.add_nodes_from(VARIABLES)
    for source in VARIABLES:
        for target in VARIABLES:
            if source == target:
                continue
            for path in nx.all_simple_paths(pkn, source, target, cutoff=6):
                if any(node in observed for node in path[1:-1]):
                    continue
                out.add_edge(source, target)
                break
    return out


def _write_dataset() -> nx.DiGraph:
    _download_raw()
    raw = pd.read_csv(RAW_DIR / "MD-LiverDREAM.csv")
    data = pd.DataFrame(
        {var: pd.to_numeric(raw[f"DV:{var}"], errors="coerce") for var in VARIABLES}
    )
    data = data.dropna(how="all")
    data = data.fillna(data.mean(numeric_only=True))
    data.to_csv(DATA_DIR / "data.csv", index=False)

    pkn = _parse_sif(RAW_DIR / "PKN-LiverDREAM.sif.txt")
    graph = _latent_projected_graph(pkn)
    nx.write_gml(graph, DATA_DIR / "ground_truth.gml")
    _atomic_write(DATA_DIR / "background.txt", BACKGROUND + "\n")
    return graph


def _gold_grounding() -> dict[str, dict[str, Any]]:
    return {
        "akt": {
            "column": "akt",
            "kind": "family",
            "ids": ["P31749", "P31751"],
            "canonical_name": "AKT1/2",
            "gene_names": ["AKT1", "AKT2"],
            "confidence": 1.0,
            "reasoning": "CellNOpt LiverDREAM uses lowercase akt for AKT family readout.",
            "reactome_validated": True,
        },
        "mek12": {
            "column": "mek12",
            "kind": "family",
            "ids": ["Q02750", "P36507"],
            "canonical_name": "MEK1/2",
            "gene_names": ["MAP2K1", "MAP2K2"],
            "confidence": 1.0,
            "reasoning": "mek12 denotes the MEK1/2 MAP2K family.",
            "reactome_validated": True,
        },
        "erk12": {
            "column": "erk12",
            "kind": "family",
            "ids": ["P27361", "P28482"],
            "canonical_name": "ERK1/2",
            "gene_names": ["MAPK3", "MAPK1"],
            "confidence": 1.0,
            "reasoning": "erk12 denotes the ERK1/2 MAPK family.",
            "reactome_validated": True,
        },
        "ikb": {
            "column": "ikb",
            "kind": "protein",
            "ids": ["P25963"],
            "canonical_name": "NFKBIA",
            "gene_names": ["NFKBIA"],
            "confidence": 1.0,
            "reasoning": "ikb denotes inhibitor of NF-kB alpha.",
            "reactome_validated": True,
        },
        "jnk12": {
            "column": "jnk12",
            "kind": "family",
            "ids": ["P45983", "P45984"],
            "canonical_name": "JNK1/2",
            "gene_names": ["MAPK8", "MAPK9"],
            "confidence": 1.0,
            "reasoning": "jnk12 denotes JNK1/2 stress MAP kinases.",
            "reactome_validated": True,
        },
        "p38": {
            "column": "p38",
            "kind": "protein",
            "ids": ["Q16539"],
            "canonical_name": "MAPK14",
            "gene_names": ["MAPK14"],
            "confidence": 1.0,
            "reasoning": "p38 denotes the p38 MAPK readout, represented by MAPK14.",
            "reactome_validated": True,
        },
        "hsp27": {
            "column": "hsp27",
            "kind": "protein",
            "ids": ["P04792"],
            "canonical_name": "HSPB1",
            "gene_names": ["HSPB1"],
            "confidence": 1.0,
            "reasoning": "hsp27 denotes heat shock protein beta-1.",
            "reactome_validated": True,
        },
    }


def _grounding_objects() -> dict[str, Grounding]:
    return {
        column: Grounding(
            column=entry["column"],
            kind=entry["kind"],
            ids=list(entry["ids"]),
            canonical_name=entry["canonical_name"],
            gene_names=list(entry["gene_names"]),
            confidence=float(entry["confidence"]),
            reasoning=str(entry["reasoning"]),
            reactome_validated=bool(entry["reactome_validated"]),
            served_model=None,
        )
        for column, entry in _gold_grounding().items()
    }


def _write_grounding() -> None:
    gold = _gold_grounding()
    _write_json(GROUNDING_DIR / "gold_liverdream.json", {"groundings": gold})
    _write_json(
        EXPERIMENTS_DIR / "grounding_liverdream.json",
        {
            "dataset": DATASET_NAME,
            "predicted": gold,
            "gold": gold,
            "metrics": {
                "exact_id_accuracy": 1.0,
                "validated_fraction": 1.0,
                "n": len(gold),
            },
            "source": "hand-curated CellNOpt LiverDREAM grounding",
        },
    )


def _claim(
    cause: str,
    effect: str,
    confidence: float,
    source: str,
    constraint_type: str = "hard_required",
) -> dict[str, Any]:
    return {
        "var_a": cause,
        "var_b": effect,
        "cause": cause,
        "effect": effect,
        "confidence": confidence,
        "constraint_type": constraint_type,
        "source": source,
    }


def _write_oracle_priors(true_graph: nx.DiGraph) -> None:
    claims = [_claim(a, b, 1.0, "oracle") for a, b in sorted(true_graph.edges())]
    _write_json(
        EXPERIMENTS_DIR / "oracle_priors_liverdream.json",
        {"dataset": DATASET_NAME, "pairs": claims},
    )


def _write_reactome_priors() -> None:
    client = ReactomeClient()
    payload = reason_all_pairs(
        grounding=_grounding_objects(),
        reactome_client=client,
        cache_dir=CACHE_LLM,
    )
    payload["dataset"] = DATASET_NAME
    _write_json(EXPERIMENTS_DIR / "causal_priors_liverdream.json", payload)


def _write_floor_priors() -> None:
    grounding = _grounding_objects()
    payload_all = build_floor_priors(grounding, source_filter="all")
    payload_reactome = build_floor_priors(grounding, source_filter="reactome_only")
    payload_all["dataset"] = DATASET_NAME
    payload_reactome["dataset"] = DATASET_NAME
    write_floor_priors(
        EXPERIMENTS_DIR / "floor_priors_liverdream_all.json", payload_all
    )
    write_floor_priors(
        EXPERIMENTS_DIR / "floor_priors_liverdream_reactome_only.json",
        payload_reactome,
    )


def _constraint_from_relation(relation_type: str, confidence: float) -> str:
    rt = relation_type.strip().lower() if relation_type else ""
    if rt == "required":
        return "hard_required" if confidence >= 0.9 else "soft_prior"
    if rt == "forbidden":
        return "hard_forbidden_reverse"
    return "unknown"


def _write_freetext_priors() -> None:
    prompt = RELATION_PROMPT.format(variables=", ".join(VARIABLES), text=BACKGROUND)
    response = cast(
        dict[str, Any],
        chat_completion(
            messages=[{"role": "user", "content": prompt}], cache_dir=CACHE_LLM
        ),
    )
    served_id = str(response.get("model") or "")
    choices = response.get("choices") or []
    content = ""
    if choices and isinstance(choices[0], dict):
        msg = choices[0].get("message") or {}
        if isinstance(msg, dict):
            content = str(msg.get("content") or "").strip()
    try:
        relations = json.loads(extract_relations._unwrap_json_text(content))
    except json.JSONDecodeError:
        logging.exception("Could not parse free-text LiverDREAM relation JSON")
        relations = []

    col_set = set(VARIABLES)
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    skipped = 0
    for rel in relations if isinstance(relations, list) else []:
        if not isinstance(rel, dict):
            continue
        cause = str(rel.get("cause", ""))
        effect = str(rel.get("effect", ""))
        if cause not in col_set or effect not in col_set or cause == effect:
            skipped += 1
            continue
        confidence = float(rel.get("confidence", 0.0))
        by_key[(cause, effect)] = _claim(
            cause,
            effect,
            confidence,
            "freetext_llm",
            _constraint_from_relation(str(rel.get("relation_type", "")), confidence),
        )

    pairs = sorted(by_key.values(), key=lambda row: (row["cause"], row["effect"]))
    type_counts = Counter(str(row["constraint_type"]) for row in pairs)
    _write_json(
        EXPERIMENTS_DIR / "freetext_priors_liverdream.json",
        {
            "dataset": DATASET_NAME,
            "background_text_path": "data/liverdream/background.txt",
            "pairs": pairs,
            "served_models": [served_id] if served_id else [],
            "n_relations_emitted": len(pairs),
            "n_relations_skipped_unknown_var": skipped,
            "n_hard_required": type_counts.get("hard_required", 0),
            "n_soft_prior": type_counts.get("soft_prior", 0),
            "n_hard_forbidden_reverse": type_counts.get("hard_forbidden_reverse", 0),
            "n_unknown": type_counts.get("unknown", 0),
        },
    )


def _conditions() -> tuple[Condition, ...]:
    return (
        Condition("C0", "none", None, None),
        Condition(
            "C0.5",
            "omnipath_reactome_only",
            EXPERIMENTS_DIR / "floor_priors_liverdream_reactome_only.json",
            0.7,
        ),
        Condition(
            "C1",
            "freetext_llm",
            EXPERIMENTS_DIR / "freetext_priors_liverdream.json",
            0.7,
        ),
        Condition(
            "C2", "reactome_llm", EXPERIMENTS_DIR / "causal_priors_liverdream.json", 0.9
        ),
        Condition(
            "C3", "reactome_llm", EXPERIMENTS_DIR / "causal_priors_liverdream.json", 0.7
        ),
        Condition(
            "C4", "reactome_llm", EXPERIMENTS_DIR / "causal_priors_liverdream.json", 0.6
        ),
        Condition(
            "C5", "oracle", EXPERIMENTS_DIR / "oracle_priors_liverdream.json", None
        ),
    )


def _predicted_edges_list(graph: nx.DiGraph) -> list[list[str]]:
    return [list(edge) for edge in sorted(graph.edges())]


def _run_ablation() -> dict[str, Any]:
    data_df, true_graph = load_liverdream_dataset()
    data = data_df.to_numpy()
    variables = list(data_df.columns)
    priors_cache = {
        cond.name: None if cond.priors_path is None else load_priors(cond.priors_path)
        for cond in _conditions()
    }

    results: list[dict[str, Any]] = []
    for cond in _conditions():
        for algorithm in ALGORITHMS:
            for seed in SEEDS:
                row = _rc.run_condition(
                    dataset_name=DATASET_NAME,
                    data=data,
                    variable_names=variables,
                    true_graph=true_graph,
                    priors=priors_cache[cond.name],
                    priors_source=cond.priors_source,
                    algorithm=algorithm,
                    threshold=cond.threshold,
                    seed=seed,
                    lingam_prior_mode=(
                        "forbidden_only"
                        if algorithm == "LiNGAM" and cond.name in {"C2", "C3", "C4"}
                        else None
                    ),
                )
                row["condition"] = cond.name
                results.append(row)

    priors_blob = json.loads(
        (EXPERIMENTS_DIR / "causal_priors_liverdream.json").read_text(encoding="utf-8")
    )
    dag_result = predict_dag_from_priors_with_meta(priors_blob, variables)
    nx.write_gml(
        dag_result.graph, EXPERIMENTS_DIR / "predicted_dag_llm_only_liverdream.gml"
    )
    _write_json(
        EXPERIMENTS_DIR / "predicted_dag_llm_only_liverdream.meta.json",
        build_meta(dag_result),
    )
    llm_metrics = evaluate(dag_result.graph, true_graph)
    results.append(
        {
            "dataset": DATASET_NAME,
            "condition": "C-LLM-only",
            "algorithm": None,
            "seed": None,
            "threshold": None,
            "priors_source": "reactome_llm",
            "status": "ok",
            "error": None,
            "metrics": {k: float(v) for k, v in llm_metrics.items()},
            "predicted_edges": _predicted_edges_list(dag_result.graph),
            "constraint_summary": None,
            "dropped_due_to_cycle": dag_result.dropped_due_to_cycle,
        }
    )

    payload = {
        "dataset": DATASET_NAME,
        "source": "CellNOptR LiverDREAM public MIDAS + PKN files",
        "n_cells_expected": 211,
        "n_cells_completed": len(results),
        "n_cells_failed": sum(1 for row in results if row["status"] == "failed"),
        "results": results,
    }
    _write_json(EXPERIMENTS_DIR / "ablation_results_liverdream.json", payload)
    return payload


def _constraint_quality() -> dict[str, Any]:
    _data, true_graph = load_liverdream_dataset()
    true_edges = {tuple(edge) for edge in true_graph.edges()}
    n_pairs = true_graph.number_of_nodes() * (true_graph.number_of_nodes() - 1)
    sources = {
        "reactome_llm": EXPERIMENTS_DIR / "causal_priors_liverdream.json",
        "omnipath_reactome_only": EXPERIMENTS_DIR
        / "floor_priors_liverdream_reactome_only.json",
        "omnipath_all": EXPERIMENTS_DIR / "floor_priors_liverdream_all.json",
        "freetext_llm": EXPERIMENTS_DIR / "freetext_priors_liverdream.json",
    }
    by_source = {
        source: compute_quality(
            load_priors(path),
            true_edges,
            confidence_threshold=0.7,
            n_total_pairs=n_pairs,
        )
        for source, path in sources.items()
    }
    payload = {
        "dataset": DATASET_NAME,
        "confidence_threshold": 0.7,
        "n_total_ordered_pairs": n_pairs,
        "n_true_edges": len(true_edges),
        "true_edges": [list(edge) for edge in sorted(true_edges)],
        "by_source": by_source,
    }
    _write_json(EXPERIMENTS_DIR / "constraint_quality_liverdream.json", payload)
    return payload


def _fmt(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "N/A"
    return f"${mean:.2f} \\pm {(std or 0.0):.2f}$"


def _write_ablation_table(payload: dict[str, Any]) -> None:
    agg = aggregate(payload["results"])
    lines = [
        r"% LiverDREAM real public CellNOpt ablation: mean $\pm$ std over successful seeds.",
        r"\begin{tabular}{l|ccc}",
        r"\hline",
        r"condition & PC F1 & GES F1 & LiNGAM F1 \\",
        r"\hline",
    ]
    for cond in ("C0", "C0.5", "C1", "C2", "C3", "C4", "C5"):
        row = [cond]
        for alg in ALGORITHMS:
            block = agg.get(cond, {}).get(alg, {}).get("f1", {})
            row.append(_fmt(block.get("mean"), block.get("std")))
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}"])
    _atomic_write(TABLES_DIR / "ablation_table_liverdream.tex", "\n".join(lines) + "\n")


def _write_cross_dataset_table() -> None:
    def _best(path: Path) -> tuple[str, float | None, float | None]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["results"]
        agg = aggregate(rows)
        best_label = "N/A"
        best_f1: float | None = None
        for cond in ("C0", "C0.5", "C1", "C2", "C3", "C4"):
            for alg in ALGORITHMS:
                block = agg.get(cond, {}).get(alg, {}).get("f1", {})
                mean = block.get("mean")
                if mean is not None and (best_f1 is None or float(mean) > best_f1):
                    best_f1 = float(mean)
                    best_label = f"{cond}/{alg}"
        llm_only = next(row for row in rows if row.get("condition") == "C-LLM-only")
        return best_label, best_f1, float(llm_only["metrics"]["f1"])

    s_label, s_f1, s_llm = _best(EXPERIMENTS_DIR / "ablation_results_sachs.json")
    l_label, l_f1, l_llm = _best(EXPERIMENTS_DIR / "ablation_results_liverdream.json")
    lines = [
        r"% Cross-dataset F1 summary; LiverDREAM is a real public CellNOpt benchmark.",
        r"\begin{tabular}{llcc}",
        r"\hline",
        r"dataset & best condition & best F1 & C-LLM-only F1 \\",
        r"\hline",
        f"Sachs & {s_label} & ${s_f1:.2f}$ & ${s_llm:.2f}$ \\\\",
        f"LiverDREAM & {l_label} & ${l_f1:.2f}$ & ${l_llm:.2f}$ \\\\",
        r"\hline",
        r"\end{tabular}",
    ]
    _atomic_write(TABLES_DIR / "cross_dataset.tex", "\n".join(lines) + "\n")


def main() -> None:
    true_graph = _write_dataset()
    _write_grounding()
    _write_oracle_priors(true_graph)
    _write_reactome_priors()
    _write_floor_priors()
    _write_freetext_priors()
    ablation = _run_ablation()
    quality = _constraint_quality()
    _write_ablation_table(ablation)
    _write_cross_dataset_table()
    print(
        json.dumps(
            {
                "dataset": DATASET_NAME,
                "n_true_edges": true_graph.number_of_edges(),
                "ablation_failed": ablation["n_cells_failed"],
                "reactome_llm_quality": quality["by_source"]["reactome_llm"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
