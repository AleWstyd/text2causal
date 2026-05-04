from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

from causal_discovery.run_ges import run_ges
from causal_discovery.run_lingam import run_lingam
from causal_discovery.run_pc import run_pc
from evaluation.harness import evaluate
from utils.load_data import load_sachs_dataset

RESULT_PATH = Path("experiments/baseline_sachs.json")
SEEDS = range(10)
METRICS = (
    "shd",
    "aupr",
    "precision",
    "recall",
    "f1",
    "directed_precision",
    "directed_recall",
    "directed_f1",
)


def _summarise(seed_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        values = [result[metric] for result in seed_metrics]
        summary[metric] = {
            "mean": float(mean(values)),
            "std": float(pstdev(values)),
        }
    return summary


def _print_summary(results: dict[str, dict[str, dict[str, float]]]) -> None:
    header = (
        f"{'Algorithm':<10} {'SHD':>8} {'AUPR':>8} {'P':>8} {'R':>8} "
        f"{'F1':>8} {'dP':>8} {'dR':>8} {'dF1':>8}"
    )
    print(header)
    print("-" * len(header))
    for algorithm, metrics in results.items():
        fields = [algorithm]
        for metric in METRICS:
            value = metrics[metric]
            fields.append(f"{value['mean']:.4f} +/- {value['std']:.4f}")
        print(
            f"{fields[0]:<10} {fields[1]:>8} {fields[2]:>8} {fields[3]:>8} "
            f"{fields[4]:>8} {fields[5]:>8} {fields[6]:>8} {fields[7]:>8} {fields[8]:>8}"
        )


def run_baseline() -> dict[str, dict[str, dict[str, float]]]:
    data, true_graph = load_sachs_dataset()
    variable_names = list(data.columns)
    data_matrix = data.to_numpy()
    runners = {
        "PC": run_pc,
        "GES": run_ges,
        "LiNGAM": run_lingam,
    }

    results: dict[str, dict[str, dict[str, float]]] = {}
    for algorithm, runner in runners.items():
        seed_metrics: list[dict[str, float]] = []
        for seed in SEEDS:
            random.seed(seed)
            np.random.seed(seed)
            if algorithm == "PC":
                predicted_graph, _ = runner(data_matrix, variable_names, None)
            else:
                predicted_graph = runner(data_matrix, variable_names, None)
            seed_metrics.append(evaluate(predicted_graph, true_graph))
        results[algorithm] = _summarise(seed_metrics)

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )
    return results


def main() -> None:
    results = run_baseline()
    _print_summary(results)
    print(f"\nWrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
