"""Step 6 Phase 4 — paper tables and figures from ablation + constraint-quality artefacts."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

CONDITION_ORDER: Final[tuple[str, ...]] = (
    "C0",
    "C0.5",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
)
ALGORITHMS: Final[tuple[str, ...]] = ("PC", "GES", "LiNGAM")
AGGREGATE_METRICS: Final[tuple[str, ...]] = ("shd", "f1", "directed_f1")
HEADLINE_F1_METRIC: Final[str] = "directed_f1"

GAP_CONDITIONS: Final[tuple[str, ...]] = ("C0.5", "C1", "C2", "C3", "C4")
NON_ORACLE: Final[tuple[str, ...]] = ("C0", "C0.5", "C1", "C2", "C3", "C4")
LLM_CD_CONDITIONS: Final[tuple[str, ...]] = ("C2", "C3", "C4")

CONSTRAINT_SOURCES: Final[tuple[str, ...]] = (
    "reactome_llm",
    "omnipath_reactome_only",
    "omnipath_all",
    "freetext_llm",
)

# Colour-blind-friendly distinct colours for grouped conditions (not algorithms).
_GAP_PALETTE: Final[tuple[str, ...]] = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
)

_ALG_COLOUR: Final[dict[str, str]] = {
    "PC": "tab:blue",
    "GES": "tab:orange",
    "LiNGAM": "tab:green",
}


def round2(x: float | None) -> float | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x), 2)


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean ± std per (condition, algorithm, metric); only ``status == "ok"`` rows.

    Returns a nested dict:
    ``out[condition][algorithm][metric] = {"mean": float|None, "std": float|None, "n_ok": int}``.

    Rows with ``algorithm is None`` or ``condition == "C-LLM-only"`` are ignored.
    """
    algo_rows = [
        r
        for r in results
        if r.get("algorithm") is not None and r.get("condition") != "C-LLM-only"
    ]
    pairs = {(str(r["condition"]), str(r["algorithm"])) for r in algo_rows}

    def _cond_sort_key(c: str) -> tuple[int, int, str]:
        if c in CONDITION_ORDER:
            return (0, CONDITION_ORDER.index(c), c)
        return (1, 99, c)

    buckets: dict[tuple[str, str, str], list[float]] = {}
    for row in algo_rows:
        if row.get("status") != "ok":
            continue
        metrics = row.get("metrics") or {}
        cond = str(row["condition"])
        alg = str(row["algorithm"])
        for m in AGGREGATE_METRICS:
            if m not in metrics or metrics[m] is None:
                continue
            key = (cond, alg, m)
            buckets.setdefault(key, []).append(float(metrics[m]))

    out: dict[str, Any] = {}
    for cond, alg in sorted(pairs, key=lambda t: (_cond_sort_key(t[0]), t[1])):
        out.setdefault(cond, {})
        out[cond].setdefault(alg, {})
        for m in AGGREGATE_METRICS:
            vals = buckets.get((cond, alg, m), [])
            n_ok = len(vals)
            if n_ok == 0:
                out[cond][alg][m] = {"mean": None, "std": None, "n_ok": 0}
            else:
                mean = float(statistics.mean(vals))
                std = float(statistics.stdev(vals)) if n_ok >= 2 else 0.0
                out[cond][alg][m] = {
                    "mean": mean,
                    "std": std,
                    "n_ok": n_ok,
                }
    return out


def extract_llm_only_row(results: list[dict[str, Any]]) -> dict[str, Any]:
    for row in results:
        if row.get("condition") == "C-LLM-only":
            return row
    raise ValueError("No C-LLM-only row in ablation results")


def _cell_tex(
    agg: dict[str, Any],
    condition: str,
    algorithm: str,
    metric: str,
    *,
    force_na: bool = False,
) -> str:
    if force_na:
        return r"N/A (failed)\textsuperscript{*}"
    block = agg.get(condition, {}).get(algorithm, {}).get(metric)
    if not block or block.get("mean") is None:
        return r"N/A\textsuperscript{*}"
    mean = round2(block["mean"])
    std = round2(block["std"]) if block.get("std") is not None else 0.0
    assert mean is not None and std is not None
    n_ok = int(block["n_ok"])
    mark = ""
    if n_ok < 10:
        mark = r"\textsuperscript{\dag}"
    return f"${mean:.2f} \\pm {std:.2f}${mark}"


def write_ablation_table(
    agg: dict[str, Any],
    llm_only_row: dict[str, Any],
    output_path: Path,
) -> None:
    """Write ``tabular`` only (no ``table`` environment). C-LLM-only uses dashes in non-PC columns."""
    lines: list[str] = [
        r"% Sachs ablation: mean $\pm$ std over seeds with status=ok; $\pm$ is sample std over those seeds (0 if only one successful seed).",
        r"% F1 columns are \textbf{skeleton} (undirected overlap; reversed arcs count as correct). For directed F1 see \texttt{ablation\_table\_directed.tex}.",
        r"% \textsuperscript{*} N/A: no successful runs or inapplicable cell. \textsuperscript{\dag}: fewer than 10/10 successful seeds (see caption).",
        r"\begin{tabular}{l|cc|cc|cc}",
        r"\hline",
        r" & \multicolumn{2}{c|}{PC} & \multicolumn{2}{c|}{GES} & \multicolumn{2}{c}{LiNGAM} \\",
        r" & SHD & F1 & SHD & F1 & SHD & F1 \\",
        r"\hline",
    ]

    table_rows = [c for c in CONDITION_ORDER if c != "C5"] + ["C-LLM-only", "C5"]
    for condition in table_rows:
        if condition == "C-LLM-only":
            m = llm_only_row.get("metrics") or {}
            shd = round2(float(m["shd"])) if m.get("shd") is not None else None
            f1 = round2(float(m["f1"])) if m.get("f1") is not None else None
            assert shd is not None and f1 is not None
            row = (
                r"C-LLM-only\textsuperscript{$\dagger$} & "
                rf"${shd:.2f}$ & ${f1:.2f}$ & --- & --- & --- & --- \\"
            )
            lines.append(
                row
                + r" % single DAG; no PC/GES/LiNGAM split—metrics repeated only under PC columns"
            )
            continue

        row_parts = [condition]
        for alg in ALGORITHMS:
            for metric in ("shd", "f1"):
                row_parts.append(
                    _cell_tex(
                        agg,
                        condition,
                        alg,
                        metric,
                    )
                )
        lines.append(" & ".join(row_parts) + r" \\")

    lines.append(r"\hline")
    lines.append(
        r"% C-LLM-only row: dagger marks algorithm-free DAG; values shown only under PC columns (--- elsewhere)."
    )
    lines.append(r"\end{tabular}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ablation_table_directed(
    agg: dict[str, Any],
    llm_only_row: dict[str, Any],
    output_path: Path,
) -> None:
    """Same layout as :func:`write_ablation_table` but F1 uses ``directed_f1`` (ordered arcs)."""
    lines: list[str] = [
        r"% Sachs ablation (directed F1): mean $\pm$ std over seeds with status=ok.",
        r"% Directed F1 requires matching arc orientation; see \texttt{ablation\_table.tex} for skeleton F1.",
        r"% \textsuperscript{*} N/A: no successful runs or inapplicable cell. \textsuperscript{\dag}: fewer than 10/10 successful seeds (see caption).",
        r"\begin{tabular}{l|cc|cc|cc}",
        r"\hline",
        r" & \multicolumn{2}{c|}{PC} & \multicolumn{2}{c|}{GES} & \multicolumn{2}{c}{LiNGAM} \\",
        r" & SHD & F1\textsubscript{dir} & SHD & F1\textsubscript{dir} & SHD & F1\textsubscript{dir} \\",
        r"\hline",
    ]

    table_rows = [c for c in CONDITION_ORDER if c != "C5"] + ["C-LLM-only", "C5"]
    for condition in table_rows:
        if condition == "C-LLM-only":
            m = llm_only_row.get("metrics") or {}
            shd_v = round2(float(m["shd"])) if m.get("shd") is not None else None
            d_f1 = (
                round2(float(m["directed_f1"]))
                if m.get("directed_f1") is not None
                else None
            )
            assert shd_v is not None and d_f1 is not None
            row = (
                r"C-LLM-only\textsuperscript{$\dagger$} & "
                rf"${shd_v:.2f}$ & ${d_f1:.2f}$ & --- & --- & --- & --- \\"
            )
            lines.append(
                row
                + r" % single DAG; no PC/GES/LiNGAM split—metrics repeated only under PC columns"
            )
            continue

        row_parts = [condition]
        for alg in ALGORITHMS:
            for metric in ("shd", "directed_f1"):
                row_parts.append(
                    _cell_tex(
                        agg,
                        condition,
                        alg,
                        metric,
                    )
                )
        lines.append(" & ".join(row_parts) + r" \\")

    lines.append(r"\hline")
    lines.append(
        r"% C-LLM-only row: dagger marks algorithm-free DAG; values shown only under PC columns (--- elsewhere)."
    )
    lines.append(r"\end{tabular}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_constraint_quality_table(quality: dict[str, Any], output_path: Path) -> None:
    """Build from ``constraint_quality_sachs.json`` ``by_source`` block."""
    by_source = quality["by_source"]
    lines = [
        r"% Constraint quality at $\tau=0.7$: precision/recall use forward soft+hard predictions; hallucination\_strict is reverse-of-true rate.",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"source & precision & recall & hallucination\_strict & coverage \\",
        r"\hline",
    ]
    for src in CONSTRAINT_SOURCES:
        row = by_source[src]
        p = round2(float(row["precision_forward"]))
        r_ = round2(float(row["recall_forward"]))
        h = round2(float(row["hallucination_strict"]))
        c = round2(float(row["coverage"]))
        assert None not in (p, r_, h, c)
        name = src.replace("_", r"\_")
        lines.append(f"{name} & ${p:.2f}$ & ${r_:.2f}$ & ${h:.2f}$ & ${c:.2f}$ \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_aupr_extension_table(
    results: list[dict[str, Any]], output_path: Path
) -> None:
    """Write PC-only AUPR / skeleton F1 / directed F1 snippet from ablation rows."""
    rows = [
        row
        for row in results
        if row.get("algorithm") == "PC"
        and row.get("condition") in CONDITION_ORDER
        and row.get("status") == "ok"
    ]
    lines = [
        r"% PC AUPR extension: mean $\pm$ std over successful seeds from ablation_results_sachs.json.",
        r"% F1 = skeleton (undirected); F1\textsubscript{dir} = directed arc overlap.",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"condition & AUPR & F1 & F1\textsubscript{dir} \\",
        r"\hline",
    ]
    for condition in CONDITION_ORDER:
        vals_aupr = [
            float((row.get("metrics") or {})["aupr"])
            for row in rows
            if row.get("condition") == condition
            and "aupr" in (row.get("metrics") or {})
        ]
        vals_f1 = [
            float((row.get("metrics") or {})["f1"])
            for row in rows
            if row.get("condition") == condition and "f1" in (row.get("metrics") or {})
        ]
        vals_df1 = [
            float((row.get("metrics") or {})["directed_f1"])
            for row in rows
            if row.get("condition") == condition
            and "directed_f1" in (row.get("metrics") or {})
        ]
        if not vals_aupr or not vals_f1 or not vals_df1:
            lines.append(f"{condition} & N/A & N/A & N/A \\\\")
            continue
        aupr_mean = statistics.mean(vals_aupr)
        aupr_std = statistics.stdev(vals_aupr) if len(vals_aupr) >= 2 else 0.0
        f1_mean = statistics.mean(vals_f1)
        f1_std = statistics.stdev(vals_f1) if len(vals_f1) >= 2 else 0.0
        df1_mean = statistics.mean(vals_df1)
        df1_std = statistics.stdev(vals_df1) if len(vals_df1) >= 2 else 0.0
        lines.append(
            f"{condition} & ${aupr_mean:.2f} \\pm {aupr_std:.2f}$ "
            f"& ${f1_mean:.2f} \\pm {f1_std:.2f}$ "
            f"& ${df1_mean:.2f} \\pm {df1_std:.2f}$ \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean_metric(
    agg: dict[str, Any], condition: str, algorithm: str, metric: str
) -> float | None:
    block = agg.get(condition, {}).get(algorithm, {}).get(metric)
    if not block:
        return None
    mean = block.get("mean")
    if mean is None:
        return None
    return float(mean)


def gap_closed_pct(
    f_cx: float | None, f_c0: float | None, f_c5: float | None
) -> float | None:
    if f_cx is None or f_c0 is None or f_c5 is None:
        return None
    den = f_c5 - f_c0
    if abs(den) < 1e-12:
        return None
    return round2((f_cx - f_c0) / den * 100.0)


def headline_summary(
    agg: dict[str, Any],
    llm_only_row: dict[str, Any],
    quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministic summary dict (rounded); nested structures sorted by key where relevant."""
    metrics_llm = llm_only_row.get("metrics") or {}
    sk_f1_cllm = (
        round2(float(metrics_llm["f1"])) if metrics_llm.get("f1") is not None else None
    )
    df1_cllm = (
        round2(float(metrics_llm[HEADLINE_F1_METRIC]))
        if metrics_llm.get(HEADLINE_F1_METRIC) is not None
        else None
    )
    shd_cllm = (
        round2(float(metrics_llm["shd"]))
        if metrics_llm.get("shd") is not None
        else None
    )

    best_by_alg: dict[str, Any] = {}
    gap_for_best: list[float] = []

    for alg in ALGORITHMS:
        best_cond: str | None = None
        best_df1: float | None = None
        best_sk_f1: float | None = None
        best_shd: float | None = None
        for cond in NON_ORACLE:
            f1m = _mean_metric(agg, cond, alg, HEADLINE_F1_METRIC)
            if f1m is None:
                continue
            if best_df1 is None or f1m > best_df1:
                best_df1 = f1m
                best_cond = cond
                shdm = _mean_metric(agg, cond, alg, "shd")
                best_shd = round2(shdm) if shdm is not None else None
                sk = _mean_metric(agg, cond, alg, "f1")
                best_sk_f1 = round2(sk) if sk is not None else None

        f0 = _mean_metric(agg, "C0", alg, HEADLINE_F1_METRIC)
        f5 = _mean_metric(agg, "C5", alg, HEADLINE_F1_METRIC)
        g_pct = gap_closed_pct(best_df1, f0, f5)
        if g_pct is not None:
            gap_for_best.append(float(g_pct))

        n_ok = 0
        if best_cond is not None:
            block = agg.get(best_cond, {}).get(alg, {}).get(HEADLINE_F1_METRIC)
            if block:
                n_ok = int(block.get("n_ok", 0))

        best_by_alg[alg] = {
            "condition": best_cond,
            "directed_f1": round2(best_df1) if best_df1 is not None else None,
            "f1": best_sk_f1,
            "shd": best_shd,
            "gap_closed_pct": g_pct,
            "n_ok_f1": n_ok,
        }

    gap_closed_best = max(gap_for_best) if gap_for_best else None

    cllm_gap_pcts: list[float] = []
    for alg in ALGORITHMS:
        f0 = _mean_metric(agg, "C0", alg, HEADLINE_F1_METRIC)
        f5 = _mean_metric(agg, "C5", alg, HEADLINE_F1_METRIC)
        g = gap_closed_pct(
            float(df1_cllm) if df1_cllm is not None else None,
            f0,
            f5,
        )
        if g is not None:
            cllm_gap_pcts.append(float(g))
    cllm_gap_max = max(cllm_gap_pcts) if cllm_gap_pcts else None

    oracle: dict[str, Any] = {}
    for alg in ALGORITHMS:
        df1_block = agg.get("C5", {}).get(alg, {}).get(HEADLINE_F1_METRIC, {})
        df1 = _mean_metric(agg, "C5", alg, HEADLINE_F1_METRIC)
        n_ok = int(df1_block.get("n_ok", 0)) if isinstance(df1_block, dict) else 0
        sk = _mean_metric(agg, "C5", alg, "f1")
        oracle[alg] = {
            "directed_f1": round2(df1) if df1 is not None else None,
            "f1": round2(sk) if sk is not None else None,
            "n_ok": n_ok,
            "failed": df1 is None,
        }
    best_oracle_f1 = max(
        (
            oracle[a]["directed_f1"]
            for a in ALGORITHMS
            if oracle[a]["directed_f1"] is not None
        ),
        default=None,
    )

    best_cd_f1: float | None = None
    for cond in LLM_CD_CONDITIONS:
        for alg in ALGORITHMS:
            v = _mean_metric(agg, cond, alg, HEADLINE_F1_METRIC)
            if v is None:
                continue
            if best_cd_f1 is None or v > best_cd_f1:
                best_cd_f1 = v

    delta = None
    if best_cd_f1 is not None and df1_cllm is not None:
        delta = round2(best_cd_f1 - float(df1_cllm))

    coverage_block = None
    if quality is not None and "coverage_conditional" in quality:
        coverage_block = {}
        for src in CONSTRAINT_SOURCES:
            row = quality["coverage_conditional"][src]
            coverage_block[src] = {
                "precision": round2(float(row["precision_forward"])),
                "recall": round2(float(row["recall_forward"])),
                "hallucination_strict": round2(float(row["hallucination_strict"])),
                "coverage": round2(float(row["coverage"])),
            }

    return {
        "best_condition_per_algorithm": best_by_alg,
        "gap_closed_pct": gap_closed_best,
        "c_llm_only": {
            "f1": sk_f1_cllm,
            "directed_f1": df1_cllm,
            "shd": shd_cllm,
            "gap_closed_pct_max_vs_algorithms": cllm_gap_max,
        },
        "oracle_c5": oracle,
        "oracle_best_f1": best_oracle_f1,
        "best_llm_cd_f1": round2(best_cd_f1) if best_cd_f1 is not None else None,
        "cd_vs_llm_only_delta": delta,
        "coverage_conditional": coverage_block,
    }


def figure_gap_closed(
    agg: dict[str, Any],
    llm_only_row: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=120)
    n_alg = len(ALGORITHMS)
    n_cond = len(GAP_CONDITIONS)
    x = np.arange(n_alg, dtype=float)
    total_w = 0.75
    bar_w = total_w / n_cond

    m_llm = llm_only_row.get("metrics") or {}
    f1_llm = (
        float(m_llm[HEADLINE_F1_METRIC])
        if m_llm.get(HEADLINE_F1_METRIC) is not None
        else None
    )

    for j, cond in enumerate(GAP_CONDITIONS):
        offsets = (j - (n_cond - 1) / 2) * bar_w
        heights: list[float] = []
        for alg in ALGORITHMS:
            f0 = _mean_metric(agg, "C0", alg, HEADLINE_F1_METRIC)
            f5 = _mean_metric(agg, "C5", alg, HEADLINE_F1_METRIC)
            fx = _mean_metric(agg, cond, alg, HEADLINE_F1_METRIC)
            g = gap_closed_pct(fx, f0, f5)
            heights.append(float(g) if g is not None else float("nan"))
        ax.bar(
            x + offsets,
            heights,
            width=bar_w * 0.92,
            label=cond,
            color=_GAP_PALETTE[j % len(_GAP_PALETTE)],
        )

    for i, alg in enumerate(ALGORITHMS):
        f0 = _mean_metric(agg, "C0", alg, HEADLINE_F1_METRIC)
        f5 = _mean_metric(agg, "C5", alg, HEADLINE_F1_METRIC)
        y_line = gap_closed_pct(f1_llm, f0, f5)
        if y_line is not None:
            ax.hlines(
                float(y_line),
                i - total_w / 2,
                i + total_w / 2,
                colors="#555555",
                linestyles="--",
                linewidth=1.2,
            )

    ax.axhline(
        0.0,
        color="#bbbbbb",
        linewidth=0.8,
        zorder=0,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(list(ALGORITHMS))
    ax.set_ylabel(
        r"$\%$ of $C0 \rightarrow C5$ $\mathrm{F1}_{\mathrm{dir}}$ gap closed"
    )
    ax.set_title("Sachs: gap closed by condition (non-oracle)")
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.92)
    ax.annotate(
        "dashed: C-LLM-only (same formula per algorithm)",
        xy=(0.98, 0.02),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#333333",
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"gap_closed.{ext}", bbox_inches="tight")
    plt.close(fig)


def figure_cd_vs_llm_only(
    agg: dict[str, Any],
    llm_only_row: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    m_llm = llm_only_row.get("metrics") or {}
    f1_llm = (
        float(m_llm[HEADLINE_F1_METRIC])
        if m_llm.get(HEADLINE_F1_METRIC) is not None
        else None
    )

    best_cd: float | None = None
    for cond in LLM_CD_CONDITIONS:
        for alg in ALGORITHMS:
            v = _mean_metric(agg, cond, alg, HEADLINE_F1_METRIC)
            if v is None:
                continue
            if best_cd is None or v > best_cd:
                best_cd = v

    oracle_vals = [
        _mean_metric(agg, "C5", alg, HEADLINE_F1_METRIC) for alg in ALGORITHMS
    ]
    oracle_best = max((v for v in oracle_vals if v is not None), default=None)

    labels = [
        "Best LLM+CD\n(max over C2–C4)",
        "C-LLM-only\n(single DAG)",
        "Oracle C5\n(best algorithm)",
    ]
    values = [
        float(best_cd) if best_cd is not None else float("nan"),
        float(f1_llm) if f1_llm is not None else float("nan"),
        float(oracle_best) if oracle_best is not None else float("nan"),
    ]
    colours = ["#2ca02c", "#ff7f0e", "#1f77b4"]

    fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=120)
    xpos = np.arange(len(labels))
    ax.bar(xpos, values, color=colours, width=0.62)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(r"$\mathrm{F1}_{\mathrm{dir}}$ (directed arc overlap)")
    ax.set_title("Sachs: does causal discovery add value vs LLM-only DAG?")
    ax.set_ylim(
        0.0, max(1.0, max((v for v in values if not math.isnan(v)), default=0.0) * 1.08)
    )
    for i, v in enumerate(values):
        if not math.isnan(v):
            ax.text(i, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"cd_vs_llm_only.{ext}", bbox_inches="tight")
    plt.close(fig)


def _aggregate_tau_curve(
    discovery_results: list[dict[str, Any]],
) -> dict[tuple[str, float], dict[str, Any]]:
    """Mean/std of directed F1 per (algorithm, threshold) for ok reactome_llm rows."""
    buckets: dict[tuple[str, float], list[float]] = {}
    for row in discovery_results:
        if row.get("priors_source") != "reactome_llm":
            continue
        if row.get("status") != "ok":
            continue
        metrics = row.get("metrics") or {}
        if HEADLINE_F1_METRIC not in metrics or metrics[HEADLINE_F1_METRIC] is None:
            continue
        alg = str(row["algorithm"])
        th = float(row["threshold"])
        buckets.setdefault((alg, th), []).append(float(metrics[HEADLINE_F1_METRIC]))

    out: dict[tuple[str, float], dict[str, Any]] = {}
    for key, vals in sorted(buckets.items()):
        n = len(vals)
        mean = float(statistics.mean(vals))
        std = float(statistics.stdev(vals)) if n >= 2 else 0.0
        out[key] = {"mean": mean, "std": std, "n_ok": n}
    return out


def figure_threshold_sensitivity(
    discovery_results: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tau_grid = (0.6, 0.7, 0.8, 0.9)
    curve = _aggregate_tau_curve(discovery_results)

    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=120)
    for alg in ALGORITHMS:
        means: list[float] = []
        stds: list[float] = []
        for t in tau_grid:
            b = curve.get((alg, t))
            if b:
                means.append(float(b["mean"]))
                stds.append(float(b["std"]))
            else:
                means.append(float("nan"))
                stds.append(0.0)
        colour = _ALG_COLOUR[alg]
        if alg == "LiNGAM" and all(math.isnan(m) for m in means):
            ax.axhline(
                0.0,
                color=colour,
                linestyle=":",
                linewidth=1.5,
                label="LiNGAM (failed: over-constrained)",
            )
            continue
        arr_m = np.array(means, dtype=float)
        arr_s = np.array(stds, dtype=float)
        ax.plot(tau_grid, arr_m, marker="o", label=alg, color=colour)
        low = arr_m - arr_s
        high = arr_m + arr_s
        ax.fill_between(tau_grid, low, high, color=colour, alpha=0.18)

    ax.set_xlabel(r"confidence threshold $\tau$")
    ax.set_ylabel(r"$\mathrm{F1}_{\mathrm{dir}}$ (mean ± std over seeds)")
    ax.set_title("Sachs: threshold sensitivity (reactome + LLM priors)")
    ax.set_xticks(tau_grid)
    ax.legend(loc="best", fontsize=8)
    ax.set_ylim(bottom=0.0, top=1.0)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"threshold_sensitivity.{ext}", bbox_inches="tight")
    plt.close(fig)


def figure_coverage_conditional_quality(
    quality: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    block = quality.get("coverage_conditional") or quality.get("by_source")
    metrics = ("precision_forward", "recall_forward", "hallucination_strict")
    metric_labels = ("Precision", "Recall", "Strict hallucination")
    x = np.arange(len(CONSTRAINT_SOURCES), dtype=float)
    width = 0.23

    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=120)
    for j, metric in enumerate(metrics):
        vals = [float(block[src][metric]) for src in CONSTRAINT_SOURCES]
        ax.bar(
            x + (j - 1) * width,
            vals,
            width=width,
            label=metric_labels[j],
            color=_GAP_PALETTE[j % len(_GAP_PALETTE)],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [src.replace("_", "\n") for src in CONSTRAINT_SOURCES],
        fontsize=8,
    )
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Rate on Reactome-covered pair subset")
    ax.set_title("Sachs: constraint quality conditional on Reactome evidence")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(
            output_dir / f"coverage_conditional_quality.{ext}", bbox_inches="tight"
        )
    plt.close(fig)


def _signed(delta: float | None) -> str:
    if delta is None:
        return "N/A"
    return f"{delta:+.2f}"


def _fmt_headline_block(s: dict[str, Any]) -> str:
    lines = [
        "=== Step 6 Sachs Ablation Headline ===",
        "Best non-oracle condition per algorithm (by directed F1):",
    ]
    best = s["best_condition_per_algorithm"]
    for alg in ALGORITHMS:
        row = best[alg]
        cond = row["condition"]
        f1v = row["directed_f1"]
        shdv = row["shd"]
        g = row["gap_closed_pct"]
        if cond is None or f1v is None:
            lines.append(f"  {alg}: N/A (no successful runs)")
            continue
        shd_s = f"{shdv:.2f}" if shdv is not None else "N/A"
        if g is None:
            lines.append(
                f"  {alg}: {cond}, F1_dir={f1v:.2f}, SHD={shd_s} (gap closed: N/A)"
            )
        else:
            lines.append(
                f"  {alg}: {cond}, F1_dir={f1v:.2f}, SHD={shd_s} (gap closed: {g:.2f}%)"
            )

    cllm = s["c_llm_only"]
    gcl = cllm["gap_closed_pct_max_vs_algorithms"]
    df1_c = cllm.get("directed_f1")
    sk1_c = cllm.get("f1")
    df1_s = f"{df1_c:.2f}" if df1_c is not None else "N/A"
    sk1_s = f"{sk1_c:.2f}" if sk1_c is not None else "N/A"
    shd_c = cllm.get("shd")
    shd_line = f"{float(shd_c):.2f}" if shd_c is not None else "N/A"
    lines.append(
        f"C-LLM-only: F1_dir={df1_s}, F1_skel={sk1_s}, SHD={shd_line}"
        + (f" (gap closed: {gcl:.2f}%)" if gcl is not None else " (gap closed: N/A)")
    )

    orch = s["oracle_c5"]
    o_parts = []
    for alg in ALGORITHMS:
        o = orch[alg]
        dfv = o["directed_f1"]
        if o["failed"] or dfv is None:
            o_parts.append(f"{alg} F1_dir=N/A (failed)")
        else:
            o_parts.append(f"{alg} F1_dir={dfv:.2f}")
    lines.append("Oracle (C5) ceiling: " + ", ".join(o_parts))

    d = s["cd_vs_llm_only_delta"]
    lines.append(
        "CD-vs-LLM-only delta (best LLM+CD F1_dir - C-LLM-only F1_dir) = " + _signed(d)
    )
    coverage = s.get("coverage_conditional")
    if coverage:
        reactome = coverage["reactome_llm"]
        omnipath = coverage["omnipath_all"]
        lines.append(
            "Coverage-conditional quality: "
            f"Reactome+LLM P={reactome['precision']:.2f}, "
            f"R={reactome['recall']:.2f}, H={reactome['hallucination_strict']:.2f}; "
            f"OmniPath-all P={omnipath['precision']:.2f}, "
            f"R={omnipath['recall']:.2f}, H={omnipath['hallucination_strict']:.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    plt.rcParams.update(
        {
            "figure.max_open_warning": 0,
            "savefig.bbox": "tight",
        }
    )

    ablation_path = REPO_ROOT / "experiments" / "ablation_results_sachs.json"
    discovery_path = REPO_ROOT / "experiments" / "discovery_results_sachs.json"
    quality_path = REPO_ROOT / "experiments" / "constraint_quality_sachs.json"

    ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
    results = ablation["results"]
    agg = aggregate(results)
    llm_row = extract_llm_only_row(results)

    discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
    disc_results = discovery["results"]

    quality = json.loads(quality_path.read_text(encoding="utf-8"))

    tables_dir = REPO_ROOT / "tables"
    figures_dir = REPO_ROOT / "figures"
    write_ablation_table(agg, llm_row, tables_dir / "ablation_table.tex")
    write_ablation_table_directed(
        agg, llm_row, tables_dir / "ablation_table_directed.tex"
    )
    write_constraint_quality_table(quality, tables_dir / "constraint_quality.tex")
    write_aupr_extension_table(results, tables_dir / "aupr_extension.tex")

    figure_gap_closed(agg, llm_row, figures_dir)
    figure_cd_vs_llm_only(agg, llm_row, figures_dir)
    figure_threshold_sensitivity(disc_results, figures_dir)
    figure_coverage_conditional_quality(quality, figures_dir)

    summary = headline_summary(agg, llm_row, quality)
    print(_fmt_headline_block(summary))
    print(
        "Wrote tables/ablation_table.tex, tables/ablation_table_directed.tex, "
        "tables/constraint_quality.tex, tables/aupr_extension.tex\n"
        "Wrote figures/gap_closed.{pdf,png}, figures/cd_vs_llm_only.{pdf,png}, "
        "figures/threshold_sensitivity.{pdf,png}, "
        "figures/coverage_conditional_quality.{pdf,png}"
    )


if __name__ == "__main__":
    main()
