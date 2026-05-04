"""Step 6.5 A2 — diagnose why PC does not move under priors on Sachs."""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path
from typing import Any, Final

from causal_discovery.run_pc import run_pc
from constraints.constraint_builder import ConstraintBuilder, load_priors
from evaluation.harness import evaluate
from utils.load_data import load_sachs_dataset

ABLATION_PATH: Final[Path] = Path("experiments/ablation_results_sachs.json")
ORACLE_PRIORS_PATH: Final[Path] = Path("experiments/oracle_priors_sachs.json")
OUTPUT_PATH: Final[Path] = Path("experiments/pc_invariance_sachs.json")
AUPR_TABLE_PATH: Final[Path] = Path("tables/aupr_extension.tex")
CONDITIONS: Final[tuple[str, ...]] = ("C0", "C2", "C3", "C4", "C5")
ALPHAS: Final[tuple[float, ...]] = (0.01, 0.05, 0.10)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _edge_set(row: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(u), str(v)) for u, v in row.get("predicted_edges") or []))


def _mean_std(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)) if len(values) >= 2 else 0.0,
        "n": len(values),
    }


def _pc_condition_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    pc_rows = [
        row
        for row in results
        if row.get("algorithm") == "PC"
        and row.get("condition") in CONDITIONS
        and row.get("status") == "ok"
    ]
    by_condition: dict[str, Any] = {}
    c0_edges: tuple[tuple[str, str], ...] | None = None

    for condition in CONDITIONS:
        rows = [row for row in pc_rows if row.get("condition") == condition]
        edge_sets = {_edge_set(row) for row in rows}
        if condition == "C0" and edge_sets:
            c0_edges = sorted(edge_sets)[0]
        by_condition[condition] = {
            "n_ok": len(rows),
            "n_unique_edge_sets": len(edge_sets),
            "metrics": {
                metric: _mean_std(
                    [
                        float((row.get("metrics") or {})[metric])
                        for row in rows
                        if metric in (row.get("metrics") or {})
                    ]
                )
                for metric in ("shd", "aupr", "precision", "recall", "f1")
            },
            "edge_set": [list(edge) for edge in sorted(edge_sets)[0]]
            if edge_sets
            else [],
        }

    if c0_edges is not None:
        for condition in CONDITIONS:
            edge_tuple = tuple(tuple(edge) for edge in by_condition[condition]["edge_set"])
            by_condition[condition]["identical_to_c0"] = edge_tuple == c0_edges

    return by_condition


def _alpha_spot_check() -> list[dict[str, Any]]:
    data_df, true_graph = load_sachs_dataset()
    variable_names = list(data_df.columns)
    data_matrix = data_df.to_numpy()
    oracle = load_priors(ORACLE_PRIORS_PATH)
    oracle_pk = ConstraintBuilder(oracle, variable_names, 0.5).to_prior_knowledge()

    rows: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        c0 = run_pc(data_matrix, variable_names, None, alpha=alpha)
        c5 = run_pc(data_matrix, variable_names, oracle_pk, alpha=alpha)
        c0_metrics = evaluate(c0, true_graph)
        c5_metrics = evaluate(c5, true_graph)
        rows.append(
            {
                "alpha": alpha,
                "c0_edges": [list(edge) for edge in sorted(c0.edges())],
                "c5_edges": [list(edge) for edge in sorted(c5.edges())],
                "identical": set(c0.edges()) == set(c5.edges()),
                "c0_metrics": {k: float(v) for k, v in c0_metrics.items()},
                "c5_metrics": {k: float(v) for k, v in c5_metrics.items()},
            }
        )
    return rows


def _write_aupr_table(summary: dict[str, Any], output_path: Path) -> None:
    lines = [
        r"% PC AUPR extension: mean $\pm$ std over successful seeds from ablation_results_sachs.json.",
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"condition & AUPR & F1 \\",
        r"\hline",
    ]
    for condition in CONDITIONS:
        metrics = summary[condition]["metrics"]
        aupr = metrics["aupr"]
        f1 = metrics["f1"]
        if aupr["mean"] is None or f1["mean"] is None:
            lines.append(f"{condition} & N/A & N/A \\\\")
            continue
        lines.append(
            f"{condition} & ${aupr['mean']:.2f} \\pm {aupr['std']:.2f}$ "
            f"& ${f1['mean']:.2f} \\pm {f1['std']:.2f}$ \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    _atomic_write_text(output_path, "\n".join(lines))


def diagnose_pc_invariance(
    ablation_path: Path = ABLATION_PATH,
    output_path: Path = OUTPUT_PATH,
    aupr_table_path: Path = AUPR_TABLE_PATH,
) -> dict[str, Any]:
    ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
    summary = _pc_condition_summary(list(ablation["results"]))
    alpha_rows = _alpha_spot_check()
    c0_equals_all = all(
        bool(summary[condition].get("identical_to_c0")) for condition in CONDITIONS
    )
    diagnosis = {
        "dataset": "sachs",
        "conditions": summary,
        "alpha_spot_check": alpha_rows,
        "diagnosis": {
            "canonical_pc_edges_identical_across_conditions": c0_equals_all,
            "alpha_spot_check_any_oracle_change": any(
                not bool(row["identical"]) for row in alpha_rows
            ),
        },
    }
    _atomic_write_json(output_path, diagnosis)
    _write_aupr_table(summary, aupr_table_path)
    return diagnosis


def main() -> None:
    payload = diagnose_pc_invariance()
    print(json.dumps(payload["diagnosis"], indent=2, sort_keys=True))
    print(f"Wrote {OUTPUT_PATH} and {AUPR_TABLE_PATH}")


if __name__ == "__main__":
    main()
