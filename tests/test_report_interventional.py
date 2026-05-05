"""Tests for :mod:`experiments.report_interventional`."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments import report_interventional as rpi


def _stub_obs_results() -> list[dict]:
    row = {
        "dataset": "sachs",
        "condition": "C0",
        "algorithm": "PC",
        "gold_version": "original",
        "seed": 0,
        "threshold": None,
        "priors_source": "none",
        "status": "ok",
        "metrics": {"cpdag_f1": 0.4, "shd_cpdag": 18.0},
    }
    return [row]


def _stub_int_results() -> list[dict]:
    base = {
        "dataset": "sachs_interventional",
        "condition": "C0",
        "gold_version": "original",
        "seed": 0,
        "threshold": None,
        "priors_source": "none",
        "status": "ok",
        "metrics": {"cpdag_f1": 0.55, "shd_cpdag": 14.0},
    }
    out = []
    for alg, strat in [
        ("PC", "naive"),
        ("GES", "naive"),
        ("LiNGAM", "naive"),
        ("GIES", "naive"),
        ("GIES", "gies"),
    ]:
        r = dict(base)
        r["algorithm"] = alg
        r["intervention_strategy"] = strat
        out.append(r)
    out.append(
        {
            "dataset": "sachs_interventional",
            "condition": "C-LLM-only",
            "algorithm": None,
            "gold_version": "original",
            "seed": None,
            "threshold": None,
            "priors_source": "reactome_llm",
            "intervention_strategy": "naive",
            "status": "ok",
            "metrics": {"cpdag_f1": 0.3, "shd_cpdag": 20.0},
        }
    )
    return out


class ReportInterventionalTests(unittest.TestCase):
    def test_writes_tex_tables(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            obs_p = root / "obs.json"
            int_p = root / "int.json"
            obs_p.write_text(
                json.dumps({"results": _stub_obs_results()}, indent=2),
                encoding="utf-8",
            )
            int_p.write_text(
                json.dumps({"results": _stub_int_results()}, indent=2),
                encoding="utf-8",
            )
            # Exercise writers directly with aggregates
            import experiments.report as rp

            agg_obs = rp.aggregate(_stub_obs_results())
            agg_int = rpi._aggregate_four(
                rpi._interventional_table_rows(_stub_int_results())
            )
            llm = next(r for r in _stub_int_results() if r["condition"] == "C-LLM-only")
            tdir = root / "tables"
            rpi.write_ablation_table_interventional(
                agg_int, llm, tdir / "ablation_table_interventional.tex"
            )
            rpi.write_observational_vs_interventional(
                agg_obs, agg_int, tdir / "observational_vs_interventional.tex"
            )
            self.assertTrue((tdir / "ablation_table_interventional.tex").is_file())
            self.assertTrue((tdir / "observational_vs_interventional.tex").is_file())
            tex = (tdir / "observational_vs_interventional.tex").read_text(
                encoding="utf-8"
            )
            self.assertIn("PC", tex)
            self.assertIn("GIES", tex)


if __name__ == "__main__":
    unittest.main()
