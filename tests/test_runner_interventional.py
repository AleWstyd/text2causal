"""Tests for :mod:`experiments.runner_interventional`."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from experiments import runner_interventional as ri


class RunnerInterventionalTests(unittest.TestCase):
    def test_expected_cell_count_c0_seed0_two_golds(self) -> None:
        full = ri._expand_with_gold(ri._build_interventional_matrix())
        algo, clm = ri._apply_cell_filters(
            full,
            ri._build_cllm_cells(),
            only_conditions=["C0"],
            only_seeds=[0],
            only_algorithms=None,
        )
        # Per gold: PC,GES,LiNGAM + GIES naive + GIES gies = 5; × 2 golds; C-LLM-only filtered out
        self.assertEqual(len(algo), 10)
        self.assertEqual(len(clm), 0)
        self.assertEqual(len(algo) + len(clm), 10)

    def test_atomic_incremental_write_mocked(self) -> None:
        def _fake_run(**kwargs):  # noqa: ANN003
            return {
                "dataset": ri.DATASET_NAME,
                "condition": "C0",
                "algorithm": kwargs["algorithm"],
                "gold_version": kwargs["gold_version"],
                "seed": kwargs["seed"],
                "threshold": kwargs["threshold"],
                "priors_source": kwargs["priors_source"],
                "intervention_strategy": kwargs["intervention_strategy"],
                "status": "ok",
                "error": None,
                "metrics": {
                    "cpdag_f1": 0.5,
                    "shd_cpdag": 10.0,
                    "directed_f1": 0.4,
                    "f1": 0.45,
                    "shd": 12.0,
                },
                "predicted_edges": [],
                "constraint_summary": None,
                "dropped_due_to_cycle": [],
                "pc_post_hoc_required_added": 0,
                "pc_post_hoc_dropped_due_to_cycle": [],
            }

        with TemporaryDirectory() as td:
            out = Path(td) / "ablation.json"
            with patch.object(
                ri._ric,
                "run_interventional_condition",
                side_effect=_fake_run,
            ):
                ri.run_interventional_ablation(
                    output_path=out,
                    only_conditions=["C0"],
                    only_seeds=[0],
                )
            self.assertTrue(out.is_file())
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertIn("results", payload)
            self.assertEqual(len(payload["results"]), 10)
            mtime = out.stat().st_mtime_ns
            with patch.object(
                ri._ric,
                "run_interventional_condition",
                side_effect=_fake_run,
            ):
                ri.run_interventional_ablation(
                    output_path=out,
                    only_conditions=["C0"],
                    only_seeds=[0],
                )
            self.assertEqual(out.stat().st_mtime_ns, mtime)


if __name__ == "__main__":
    unittest.main()
