"""Tests for experiments/runner.py (mocked discovery; no full Sachs sweep)."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd

from experiments import runner as r


def _fake_sachs() -> tuple[pd.DataFrame, nx.DiGraph]:
    cols = ["var_a", "var_b", "var_c"]
    df = pd.DataFrame(np.zeros((2, 3), dtype=float), columns=cols)
    graph = nx.DiGraph()
    graph.add_nodes_from(cols)
    graph.add_edge("var_a", "var_b")
    return df, graph


def _minimal_algo_row(
    *,
    condition: str,
    priors_source: str,
    algorithm: str,
    seed: int,
    threshold: float | None,
    status: str = "ok",
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
        "metrics": {
            "shd": 1.0,
            "f1": 0.5,
            "directed_precision": 0.5,
            "directed_recall": 0.5,
            "directed_f1": 0.5,
            "cpdag_precision": 0.5,
            "cpdag_recall": 0.5,
            "cpdag_f1": 0.5,
            "shd_cpdag": 1.0,
        },
        "predicted_edges": [],
        "constraint_summary": None,
        "dropped_due_to_cycle": [],
    }


class TestRunner(unittest.TestCase):
    def test_freetext_fallback_stale_row_detection(self) -> None:
        path = Path("experiments/causal_priors_sachs_with_fallback.json")
        if not path.is_file():
            self.skipTest("committed priors file missing")
        good = hashlib.sha256(path.read_bytes()).hexdigest()
        stale_row = {
            "priors_source": "reactome_llm_with_freetext_fallback",
            "notes": {"source_priors_hash": "0" * 64},
        }
        fresh_row = {
            "priors_source": "reactome_llm_with_freetext_fallback",
            "notes": {"source_priors_hash": good},
        }
        self.assertTrue(r._freetext_fallback_row_stale(stale_row, path=path))
        self.assertFalse(r._freetext_fallback_row_stale(fresh_row, path=path))
        self.assertFalse(
            r._freetext_fallback_row_stale(
                {"priors_source": "reactome_llm", "notes": {}},
                path=path,
            )
        )

    def test_idempotency_skips_run_condition(self) -> None:
        existing = _minimal_algo_row(
            condition="C0",
            priors_source="none",
            algorithm="PC",
            seed=0,
            threshold=None,
        )
        payload = {
            "dataset": "sachs",
            "matrix": r.MATRIX_ID,
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
            with patch.object(r._rc, "run_condition") as mock_rc:
                r.run_ablation(
                    output_path=outp,
                    only_conditions=["C0"],
                    only_algorithms=["PC"],
                    only_seeds=[0],
                )
                mock_rc.assert_not_called()
            after = json.loads(outp.read_text(encoding="utf-8"))
            self.assertEqual(before, after)

    def test_cllm_row_construction(self) -> None:
        gml = """graph [
  directed 1
  node [
    id 0
    label "a"
  ]
  node [
    id 1
    label "b"
  ]
  node [
    id 2
    label "c"
  ]
  edge [
    source 0
    target 1
  ]
  edge [
    source 1
    target 2
  ]
]
"""
        meta = {
            "threshold": 0.7,
            "source_priors_hash": "deadbeef",
        }
        true_g = nx.DiGraph()
        true_g.add_nodes_from(["a", "b", "c"])
        true_g.add_edge("a", "b")

        with TemporaryDirectory() as td:
            p = Path(td)
            gp = p / "tiny.gml"
            mp = p / "meta.json"
            gp.write_text(gml, encoding="utf-8")
            mp.write_text(json.dumps(meta), encoding="utf-8")

            row = r._run_cllm_cell(true_graph=true_g, gml_path=gp, meta_path=mp)
            self.assertEqual(row["condition"], "C-LLM-only")
            self.assertIsNone(row["algorithm"])
            self.assertIsNone(row["seed"])
            self.assertEqual(row["status"], "ok")
            self.assertIn("metrics", row)
            self.assertAlmostEqual(row["metrics"]["precision"], 0.5)
            self.assertGreater(len(row["predicted_edges"]), 0)
            notes = row["notes"]
            self.assertEqual(notes["n_nodes"], 3)
            self.assertEqual(notes["n_edges"], 2)
            self.assertEqual(notes["threshold"], 0.7)
            self.assertEqual(notes["source_priors_hash"], "deadbeef")
            self.assertIn("tiny.gml", notes["predicted_dag_path"])

    def test_cllm_idempotency(self) -> None:
        row = {
            "dataset": "sachs",
            "condition": "C-LLM-only",
            "algorithm": None,
            "seed": None,
            "threshold": None,
            "priors_source": "reactome_llm",
            "status": "ok",
            "error": None,
            "metrics": {
                "shd": 0.0,
                "directed_precision": 0.0,
                "directed_recall": 0.0,
                "directed_f1": 0.0,
                "cpdag_precision": 0.0,
                "cpdag_recall": 0.0,
                "cpdag_f1": 0.0,
                "shd_cpdag": 0.0,
            },
            "predicted_edges": [["a", "b"]],
            "constraint_summary": None,
            "dropped_due_to_cycle": [],
            "notes": {},
        }
        payload = {
            "dataset": "sachs",
            "matrix": r.MATRIX_ID,
            "n_cells_expected": 1,
            "n_cells_completed": 1,
            "n_cells_failed": 0,
            "ges_total_dropped_due_to_cycle": 0,
            "results": [row],
        }
        with TemporaryDirectory() as td:
            outp = Path(td) / "ab.json"
            outp.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            before = json.loads(outp.read_text(encoding="utf-8"))
            with patch.object(r, "_run_cllm_cell") as mock_cllm:
                r.run_ablation(
                    output_path=outp,
                    only_conditions=["C-LLM-only"],
                )
                mock_cllm.assert_not_called()
            after = json.loads(outp.read_text(encoding="utf-8"))
            self.assertEqual(before, after)

    def test_atomic_write_preserves_prior_state_on_failure(self) -> None:
        first = _minimal_algo_row(
            condition="C0",
            priors_source="none",
            algorithm="PC",
            seed=0,
            threshold=None,
        )
        payload = {
            "dataset": "sachs",
            "matrix": r.MATRIX_ID,
            "n_cells_expected": 2,
            "n_cells_completed": 1,
            "n_cells_failed": 0,
            "ges_total_dropped_due_to_cycle": 0,
            "results": [first],
        }
        with TemporaryDirectory() as td:
            outp = Path(td) / "out.json"
            outp.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            def _boom(*args: object, **kwargs: object) -> dict:
                raise RuntimeError("simulated failure")

            with (
                patch.object(r._rc, "run_condition", side_effect=_boom),
                patch.object(r, "load_sachs_dataset", return_value=_fake_sachs()),
                patch.object(r, "load_priors", return_value=[]),
            ):
                with self.assertRaises(RuntimeError):
                    r.run_ablation(
                        output_path=outp,
                        only_conditions=["C0"],
                        only_algorithms=["PC"],
                        only_seeds=[0, 1],
                    )

            reloaded = json.loads(outp.read_text(encoding="utf-8"))
            self.assertEqual(len(reloaded["results"]), 1)
            self.assertEqual(reloaded["results"][0]["seed"], 0)

    def test_sort_key_puts_none_algorithm_last_within_condition(self) -> None:
        rows = [
            {
                "condition": "Q",
                "priors_source": "x",
                "algorithm": None,
                "threshold": None,
                "seed": None,
            },
            {
                "condition": "Q",
                "priors_source": "x",
                "algorithm": "PC",
                "threshold": 0.7,
                "seed": 0,
            },
            {
                "condition": "Q",
                "priors_source": "x",
                "algorithm": "GES",
                "threshold": 0.7,
                "seed": 0,
            },
        ]
        sorted_rows = sorted(rows, key=r.result_sort_key)
        self.assertEqual(
            [row["algorithm"] for row in sorted_rows],
            ["GES", "PC", None],
        )


if __name__ == "__main__":
    unittest.main()
