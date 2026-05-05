"""Tests for experiments/run_discovery_sachs.py (mocked, no Sachs execution)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd

from experiments import run_discovery_sachs as rds


def _fake_sachs() -> tuple[pd.DataFrame, nx.DiGraph]:
    cols = ["var_a", "var_b", "var_c"]
    df = pd.DataFrame(np.zeros((2, 3), dtype=float), columns=cols)
    graph = nx.DiGraph()
    graph.add_nodes_from(cols)
    return df, graph


def _minimal_result(
    *,
    condition: str,
    priors_source: str,
    algorithm: str,
    seed: int,
    threshold: float | None,
    status: str = "ok",
    shd: float = 3.0,
) -> dict:
    return {
        "dataset": "sachs",
        "condition": condition,
        "algorithm": algorithm,
        "seed": seed,
        "threshold": threshold,
        "priors_source": priors_source,
        "status": status,
        "error": None,
        "metrics": (
            {
                "shd": shd,
                "directed_precision": 0.0,
                "directed_recall": 0.0,
                "directed_f1": 0.0,
                "cpdag_precision": 0.0,
                "cpdag_recall": 0.0,
                "cpdag_f1": 0.0,
                "shd_cpdag": shd,
            }
            if status == "ok"
            else None
        ),
        "predicted_edges": [],
        "constraint_summary": None,
        "dropped_due_to_cycle": [],
    }


class TestRunDiscoverySachs(unittest.TestCase):
    def test_full_matrix_cell_count(self) -> None:
        self.assertEqual(len(rds._build_cell_matrix()), 420)

    def test_tuple_key_none_threshold_vs_float_same_priors_source(self) -> None:
        """Resumability keys must not conflate missing threshold with a numeric sweep value."""
        none_thr = rds.row_key_from_spec(rds._CellSpec("reactome_llm", None, "PC", 0))
        numeric = rds.row_key_from_spec(rds._CellSpec("reactome_llm", 0.7, "PC", 0))
        self.assertNotEqual(none_thr, numeric)
        self.assertEqual(none_thr[3], "none")
        self.assertEqual(numeric[3], "0.70")

    def test_idempotency_skips_run_condition_and_preserves_json(self) -> None:
        existing = _minimal_result(
            condition="C0",
            priors_source="none",
            algorithm="PC",
            seed=0,
            threshold=None,
            shd=9.0,
        )
        payload = {
            "dataset": "sachs",
            "n_cells_expected": 1,
            "n_cells_completed": 1,
            "n_cells_failed": 0,
            "ges_total_dropped_due_to_cycle": 0,
            "results": [existing],
        }
        with TemporaryDirectory() as td:
            outp = Path(td) / "out.json"
            outp.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            before = json.loads(outp.read_text(encoding="utf-8"))
            with patch.object(rds, "run_condition") as mock_rc:
                rds.run_sweep(
                    output_path=outp,
                    only_conditions=["none"],
                    only_algorithms=["PC"],
                    only_seeds=[0],
                )
                mock_rc.assert_not_called()
            after = json.loads(outp.read_text(encoding="utf-8"))
            self.assertEqual(before, after)

    def test_append_flushes_once_per_new_cell(self) -> None:
        calls: list[int] = []
        orig_write = rds._atomic_write_json

        def track_write(path: Path, payload: dict) -> None:
            calls.append(len(payload["results"]))
            orig_write(path, payload)

        rows = [
            _minimal_result(
                condition="C0",
                priors_source="none",
                algorithm="PC",
                seed=i,
                threshold=None,
                shd=float(i),
            )
            for i in (0, 1, 2)
        ]
        with TemporaryDirectory() as td:
            outp = Path(td) / "out.json"
            with (
                patch.object(rds, "load_sachs_dataset", side_effect=_fake_sachs),
                patch.object(rds, "run_condition", side_effect=rows),
                patch.object(rds, "_atomic_write_json", side_effect=track_write),
            ):
                rds.run_sweep(
                    output_path=outp,
                    only_conditions=["none"],
                    only_algorithms=["PC"],
                    only_seeds=[0, 1, 2],
                )
            self.assertEqual(calls, [1, 2, 3])
            final = json.loads(outp.read_text(encoding="utf-8"))
            self.assertEqual(len(final["results"]), 3)
            seeds = sorted(r["seed"] for r in final["results"])
            self.assertEqual(seeds, [0, 1, 2])

    def test_atomic_write_survives_run_condition_raise_mid_sweep(self) -> None:
        rows = [
            _minimal_result(
                condition="C0",
                priors_source="none",
                algorithm="PC",
                seed=i,
                threshold=None,
                shd=float(i),
            )
            for i in (0, 1)
        ]
        with TemporaryDirectory() as td:
            outp = Path(td) / "out.json"
            with (
                patch.object(rds, "load_sachs_dataset", side_effect=_fake_sachs),
                patch.object(
                    rds,
                    "run_condition",
                    side_effect=[*rows, RuntimeError("simulated failure")],
                ),
            ):
                with self.assertRaises(RuntimeError):
                    rds.run_sweep(
                        output_path=outp,
                        only_conditions=["none"],
                        only_algorithms=["PC"],
                        only_seeds=[0, 1, 2],
                    )
            data = json.loads(outp.read_text(encoding="utf-8"))
            self.assertEqual(len(data["results"]), 2)
            seeds = {r["seed"] for r in data["results"]}
            self.assertEqual(seeds, {0, 1})


if __name__ == "__main__":
    unittest.main()
