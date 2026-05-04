"""Tests for ``experiments.report`` Step 6 Phase 4 reporting."""

from __future__ import annotations

import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from experiments import report


def _ok_row(
    *,
    condition: str,
    algorithm: str,
    seed: int,
    f1: float,
    shd: float,
    directed_f1: float | None = None,
) -> dict:
    df1 = float(f1) if directed_f1 is None else float(directed_f1)
    return {
        "condition": condition,
        "algorithm": algorithm,
        "seed": seed,
        "status": "ok",
        "metrics": {"f1": f1, "directed_f1": df1, "shd": shd},
    }


def _fail_row(*, condition: str, algorithm: str, seed: int) -> dict:
    return {
        "condition": condition,
        "algorithm": algorithm,
        "seed": seed,
        "status": "failed",
        "metrics": None,
    }


class TestAggregate(unittest.TestCase):
    def test_mean_std_over_ok_seeds(self) -> None:
        rows = [
            _ok_row(condition="C0", algorithm="PC", seed=0, f1=0.10, shd=5.0),
            _ok_row(condition="C0", algorithm="PC", seed=1, f1=0.30, shd=7.0),
            _ok_row(condition="C0", algorithm="PC", seed=2, f1=0.50, shd=9.0),
        ]
        agg = report.aggregate(rows)
        f1 = agg["C0"]["PC"]["f1"]
        self.assertEqual(f1["n_ok"], 3)
        self.assertAlmostEqual(f1["mean"], 0.30, places=6)
        self.assertAlmostEqual(f1["std"], 0.20, places=6)
        df1 = agg["C0"]["PC"]["directed_f1"]
        self.assertAlmostEqual(df1["mean"], 0.30, places=6)
        shd = agg["C0"]["PC"]["shd"]
        self.assertAlmostEqual(shd["mean"], 7.0, places=6)

    def test_excludes_failed_status(self) -> None:
        rows = [
            _ok_row(condition="C1", algorithm="GES", seed=0, f1=0.40, shd=10.0),
            _fail_row(condition="C1", algorithm="GES", seed=1),
            _ok_row(condition="C1", algorithm="GES", seed=2, f1=0.60, shd=12.0),
        ]
        agg = report.aggregate(rows)
        f1 = agg["C1"]["GES"]["f1"]
        self.assertEqual(f1["n_ok"], 2)
        self.assertAlmostEqual(f1["mean"], 0.50, places=6)

    def test_all_failed_returns_null_stats(self) -> None:
        rows = [_fail_row(condition="C2", algorithm="LiNGAM", seed=i) for i in range(3)]
        agg = report.aggregate(rows)
        f1 = agg["C2"]["LiNGAM"]["f1"]
        self.assertEqual(f1["n_ok"], 0)
        self.assertIsNone(f1["mean"])
        self.assertIsNone(f1["std"])
        df1 = agg["C2"]["LiNGAM"]["directed_f1"]
        self.assertEqual(df1["n_ok"], 0)
        self.assertIsNone(df1["mean"])

    def test_skips_llm_only_rows(self) -> None:
        rows = [
            _ok_row(condition="C0", algorithm="PC", seed=0, f1=0.2, shd=1.0),
            {
                "condition": "C-LLM-only",
                "algorithm": None,
                "seed": None,
                "status": "ok",
                "metrics": {"f1": 0.99, "directed_f1": 0.99, "shd": 0.0},
            },
        ]
        agg = report.aggregate(rows)
        self.assertIn("C0", agg)
        self.assertNotIn("C-LLM-only", agg)


class TestHeadlineSummary(unittest.TestCase):
    def test_gap_closed_pct_formula(self) -> None:
        agg = {
            "C0": {
                "PC": {
                    "f1": {"mean": 0.20, "std": 0.0, "n_ok": 10},
                    "directed_f1": {"mean": 0.20, "std": 0.0, "n_ok": 10},
                    "shd": {"mean": 1.0, "std": 0.0, "n_ok": 10},
                },
            },
            "C3": {
                "PC": {
                    "f1": {"mean": 0.50, "std": 0.0, "n_ok": 10},
                    "directed_f1": {"mean": 0.50, "std": 0.0, "n_ok": 10},
                    "shd": {"mean": 2.0, "std": 0.0, "n_ok": 10},
                },
            },
            "C5": {
                "PC": {
                    "f1": {"mean": 0.80, "std": 0.0, "n_ok": 10},
                    "directed_f1": {"mean": 0.80, "std": 0.0, "n_ok": 10},
                    "shd": {"mean": 3.0, "std": 0.0, "n_ok": 10},
                },
            },
        }
        for alg in ("GES", "LiNGAM"):
            agg["C0"][alg] = {
                "f1": {"mean": 0.10, "std": 0.0, "n_ok": 10},
                "directed_f1": {"mean": 0.10, "std": 0.0, "n_ok": 10},
            }
            agg["C3"][alg] = {
                "f1": {"mean": 0.15, "std": 0.0, "n_ok": 10},
                "directed_f1": {"mean": 0.15, "std": 0.0, "n_ok": 10},
            }
            agg["C5"][alg] = {
                "f1": {"mean": 0.60, "std": 0.0, "n_ok": 10},
                "directed_f1": {"mean": 0.60, "std": 0.0, "n_ok": 10},
            }

        llm = {"metrics": {"f1": 0.40, "directed_f1": 0.40, "shd": 18.0}}
        s = report.headline_summary(agg, llm)
        pc_best = s["best_condition_per_algorithm"]["PC"]
        self.assertEqual(pc_best["condition"], "C3")
        self.assertEqual(pc_best["gap_closed_pct"], 50.0)
        self.assertEqual(s["gap_closed_pct"], 50.0)

    def test_cd_vs_llm_only_delta_sign(self) -> None:
        agg: dict[str, dict] = {}
        for cond in ("C0", "C0.5", "C1", "C2", "C3", "C4", "C5"):
            agg[cond] = {}
            for alg in report.ALGORITHMS:
                agg[cond][alg] = {
                    "f1": {"mean": 0.10, "std": 0.0, "n_ok": 10},
                    "directed_f1": {"mean": 0.10, "std": 0.0, "n_ok": 10},
                    "shd": {"mean": 1.0, "std": 0.0, "n_ok": 10},
                }
        agg["C2"]["PC"]["f1"]["mean"] = 0.75
        agg["C2"]["PC"]["directed_f1"]["mean"] = 0.75
        agg["C5"]["PC"]["f1"]["mean"] = 0.90
        agg["C5"]["PC"]["directed_f1"]["mean"] = 0.90
        agg["C5"]["GES"]["f1"]["mean"] = 0.85
        agg["C5"]["GES"]["directed_f1"]["mean"] = 0.85
        agg["C5"]["LiNGAM"]["f1"]["mean"] = 0.80
        agg["C5"]["LiNGAM"]["directed_f1"]["mean"] = 0.80

        llm = {"metrics": {"f1": 0.40, "directed_f1": 0.40, "shd": 12.0}}
        s = report.headline_summary(agg, llm)
        self.assertEqual(s["cd_vs_llm_only_delta"], 0.35)

    def test_headline_summary_deterministic(self) -> None:
        rows = [
            _ok_row(condition="C0", algorithm="PC", seed=i, f1=0.5, shd=3.0)
            for i in range(5)
        ]
        agg = report.aggregate(rows)
        llm = {"metrics": {"f1": 0.4, "directed_f1": 0.4, "shd": 10.0}}
        a = report.headline_summary(agg, llm)
        b = report.headline_summary(agg, llm)
        self.assertEqual(a, b)


class TestWritersAndFigures(unittest.TestCase):
    def test_write_ablation_table_content(self) -> None:
        agg = report.aggregate(
            [
                _ok_row(condition="C0", algorithm="PC", seed=0, f1=0.5, shd=10.0),
            ]
        )
        llm = {"metrics": {"f1": 0.47, "directed_f1": 0.47, "shd": 18.0}}
        p = Path(__file__).resolve().parent / "_tmp_ablation_test.tex"
        try:
            report.write_ablation_table(agg, llm, p)
            text = p.read_text(encoding="utf-8")
            self.assertIn(r"\begin{tabular}", text)
            self.assertIn(r"\end{tabular}", text)
            self.assertIn("C-LLM-only", text)
            self.assertIn("skeleton", text.lower())
        finally:
            p.unlink(missing_ok=True)

    def test_write_ablation_table_directed_smoke(self) -> None:
        agg = report.aggregate(
            [
                _ok_row(condition="C0", algorithm="PC", seed=0, f1=0.5, shd=10.0),
            ]
        )
        llm = {"metrics": {"f1": 0.47, "directed_f1": 0.47, "shd": 18.0}}
        p = Path(__file__).resolve().parent / "_tmp_ablation_dir_test.tex"
        try:
            report.write_ablation_table_directed(agg, llm, p)
            text = p.read_text(encoding="utf-8")
            self.assertIn("F1\\textsubscript{dir}", text)
        finally:
            p.unlink(missing_ok=True)

    def test_write_constraint_quality_all_sources(self) -> None:
        quality = {
            "by_source": {
                src: {
                    "precision_forward": 0.5,
                    "recall_forward": 0.4,
                    "hallucination_strict": 0.1,
                    "coverage": 0.2,
                }
                for src in report.CONSTRAINT_SOURCES
            }
        }
        p = Path(__file__).resolve().parent / "_tmp_cq_test.tex"
        try:
            report.write_constraint_quality_table(quality, p)
            text = p.read_text(encoding="utf-8")
            for src in report.CONSTRAINT_SOURCES:
                self.assertIn(src.replace("_", r"\_"), text)
        finally:
            p.unlink(missing_ok=True)

    def test_figures_smoke(self) -> None:
        agg = report.aggregate(
            [
                _ok_row(
                    condition=c, algorithm="PC", seed=0, f1=0.4 + 0.01 * i, shd=float(i)
                )
                for i, c in enumerate(
                    ["C0", "C0.5", "C1", "C2", "C3", "C3+ft", "C4", "C5"]
                )
            ]
            + [
                _ok_row(condition=c, algorithm="GES", seed=0, f1=0.45, shd=20.0)
                for c in ["C0", "C0.5", "C1", "C2", "C3", "C3+ft", "C4", "C5"]
            ]
            + [
                _ok_row(condition=c, algorithm="LiNGAM", seed=0, f1=0.42, shd=30.0)
                for c in ["C0", "C0.5", "C1"]
            ]
            + [
                _fail_row(condition=c, algorithm="LiNGAM", seed=0)
                for c in ["C2", "C3", "C3+ft", "C4", "C5"]
            ]
        )
        llm = {"metrics": {"f1": 0.46, "directed_f1": 0.46, "shd": 17.0}}
        base = Path(__file__).resolve().parent / "_fig_smoke"
        base.mkdir(exist_ok=True)
        try:
            report.figure_gap_closed(agg, llm, base)
            report.figure_cd_vs_llm_only(agg, llm, base)
            disc = [
                {
                    "priors_source": "reactome_llm",
                    "algorithm": "PC",
                    "threshold": t,
                    "seed": 0,
                    "status": "ok",
                    "metrics": {"f1": 0.5, "directed_f1": 0.5},
                }
                for t in (0.6, 0.7, 0.8, 0.9)
            ]
            report.figure_threshold_sensitivity(disc, base)
            for name in (
                "gap_closed",
                "cd_vs_llm_only",
                "threshold_sensitivity",
            ):
                for ext in ("pdf", "png"):
                    fp = base / f"{name}.{ext}"
                    self.assertTrue(fp.is_file())
                    self.assertGreater(fp.stat().st_size, 80)
        finally:
            for fp in base.glob("*"):
                fp.unlink()
            base.rmdir()


if __name__ == "__main__":
    unittest.main()
