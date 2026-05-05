"""PR6 — observational vs interventional Sachs reporting."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Final

import experiments.report as rp

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

ALGORITHMS_INTERV: Final[tuple[str, ...]] = ("PC", "GES", "LiNGAM", "GIES")


def _interventional_table_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Interventional ablation rows: PC/GES/LiNGAM @ naive; GIES @ ``gies``."""

    out: list[dict[str, Any]] = []
    for r in results:
        if str(r.get("dataset") or "") != "sachs_interventional":
            continue
        if r.get("algorithm") is None or r.get("condition") == "C-LLM-only":
            continue
        alg = str(r["algorithm"])
        st = str(r.get("intervention_strategy") or "naive")
        if alg == "GIES" and st != "gies":
            continue
        if alg != "GIES" and st != "naive":
            continue
        out.append(r)
    return out


def _aggregate_four(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Like :func:`experiments.report.aggregate` but recognises four algorithms."""

    algo_rows = [
        r
        for r in results
        if r.get("algorithm") is not None and r.get("condition") != "C-LLM-only"
    ]
    pairs = {(str(r["condition"]), str(r["algorithm"])) for r in algo_rows}

    def _cond_sort_key(c: str) -> tuple[int, int, str]:
        if c in rp.CONDITION_ORDER:
            return (0, rp.CONDITION_ORDER.index(c), c)
        return (1, 99, c)

    metrics = ("shd", "f1", "directed_f1", "cpdag_f1", "shd_cpdag")
    buckets: dict[tuple[str, str, str], list[float]] = {}
    for row in algo_rows:
        if row.get("status") != "ok":
            continue
        m = row.get("metrics") or {}
        cond = str(row["condition"])
        alg = str(row["algorithm"])
        for met in metrics:
            if met not in m or m[met] is None:
                continue
            buckets.setdefault((cond, alg, met), []).append(float(m[met]))

    out: dict[str, Any] = {}
    for cond, alg in sorted(pairs, key=lambda t: (_cond_sort_key(t[0]), t[1])):
        out.setdefault(cond, {})
        out[cond].setdefault(alg, {})
        for met in metrics:
            vals = buckets.get((cond, alg, met), [])
            n_ok = len(vals)
            if n_ok == 0:
                out[cond][alg][met] = {"mean": None, "std": None, "n_ok": 0}
            else:
                mean = float(statistics.mean(vals))
                std = float(statistics.stdev(vals)) if n_ok >= 2 else 0.0
                out[cond][alg][met] = {
                    "mean": mean,
                    "std": std,
                    "n_ok": n_ok,
                }
    return out


def _cell_tex_four(
    agg: dict[str, Any],
    condition: str,
    algorithm: str,
    metric: str,
) -> str:
    block = agg.get(condition, {}).get(algorithm, {}).get(metric)
    if not block or block.get("mean") is None:
        return r"N/A\textsuperscript{*}"
    mean = rp.round2(block["mean"])
    std = rp.round2(block["std"]) if block.get("std") is not None else 0.0
    assert mean is not None and std is not None
    n_ok = int(block["n_ok"])
    mark = ""
    if n_ok < 10:
        mark = r"\textsuperscript{\dag}"
    return f"${mean:.2f} \\pm {std:.2f}${mark}"


def write_ablation_table_interventional(
    agg: dict[str, Any],
    llm_row: dict[str, Any],
    output_path: Path,
) -> None:
    """CPDAG / SHD-CPDAG table for the interventional Sachs matrix (four algorithms)."""

    lines: list[str] = [
        r"% Sachs interventional ablation (merged experimental contexts; see PR6).",
        r"% PC/GES/LiNGAM ignore intervention labels (naive); GIES uses native targets.",
        r"% \textsuperscript{*} N/A; \textsuperscript{\dag}: fewer than 10/10 seeds.",
        r"\begin{tabular}{l|cc|cc|cc|cc}",
        r"\hline",
        r" & \multicolumn{2}{c|}{PC} & \multicolumn{2}{c|}{GES} & "
        r"\multicolumn{2}{c|}{LiNGAM} & \multicolumn{2}{c}{GIES} \\",
        r" & SHD\textsubscript{CPDAG} & F1\textsubscript{cpdag} & "
        r"SHD\textsubscript{CPDAG} & F1\textsubscript{cpdag} & "
        r"SHD\textsubscript{CPDAG} & F1\textsubscript{cpdag} & "
        r"SHD\textsubscript{CPDAG} & F1\textsubscript{cpdag} \\",
        r"\hline",
    ]
    table_rows = [c for c in rp.CONDITION_ORDER if c != "C5"] + ["C-LLM-only", "C5"]
    for condition in table_rows:
        if condition == "C-LLM-only":
            m = llm_row.get("metrics") or {}
            shd_v = (
                rp.round2(float(m["shd_cpdag"]))
                if m.get("shd_cpdag") is not None
                else None
            )
            c_f1 = (
                rp.round2(float(m["cpdag_f1"]))
                if m.get("cpdag_f1") is not None
                else None
            )
            assert shd_v is not None and c_f1 is not None
            row = (
                r"C-LLM-only\textsuperscript{$\dagger$} & "
                rf"${shd_v:.2f}$ & ${c_f1:.2f}$ & --- & --- & --- & --- & --- & --- \\"
            )
            lines.append(row)
            continue
        row_parts = [condition]
        for alg in ALGORITHMS_INTERV:
            for metric in ("shd_cpdag", "cpdag_f1"):
                row_parts.append(_cell_tex_four(agg, condition, alg, metric))
        lines.append(" & ".join(row_parts) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_observational_vs_interventional(
    agg_obs: dict[str, Any],
    agg_int: dict[str, Any],
    output_path: Path,
) -> None:
    """Side-by-side mean CPDAG F1 (observational pooled Sachs vs interventional)."""

    lines: list[str] = [
        r"% Observational (CDT pooled rows; Step 6 JSON) vs interventional (nine contexts).",
        r"% GIES observational column is N/A (algorithm absent on observational pipeline).",
        r"\begin{tabular}{llccc}",
        r"\hline",
        r"condition & algorithm & F1\textsubscript{cpdag} (obs.) & "
        r"F1\textsubscript{cpdag} (int.) & $\Delta$ \\",
        r"\hline",
    ]
    for condition in rp.condition_table_order_with_llm():
        if condition == "C-LLM-only":
            lines.append(
                r"C-LLM-only & --- & \multicolumn{3}{c}{same DAG evaluated on both datasets} \\"
            )
            continue
        for alg in ALGORITHMS_INTERV:
            o = rp._mean_metric(agg_obs, condition, alg, rp.HEADLINE_CPDAG_METRIC)
            i = rp._mean_metric(agg_int, condition, alg, rp.HEADLINE_CPDAG_METRIC)
            if alg == "GIES":
                obs_s = r"---"
            elif o is None:
                obs_s = r"N/A"
            else:
                obs_s = f"${float(o):.2f}$"
            if i is None:
                int_s = r"N/A"
                d_s = r"N/A"
            elif alg == "GIES" or o is None:
                int_s = f"${float(i):.2f}$"
                d_s = r"N/A"
            else:
                int_s = f"${float(i):.2f}$"
                d_s = f"${float(i) - float(o):+.2f}$"
            lines.append(f"{condition} & {alg} & {obs_s} & {int_s} & {d_s} \\\\")
    lines.extend([r"\hline", r"\end{tabular}"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_interventional_headline(
    agg_obs: dict[str, Any], agg_int: dict[str, Any]
) -> None:
    """Console block: C0 / best non-oracle / C5 CPDAG F1 observational → interventional."""

    def _best_non_oracle(agg: dict[str, Any], algorithm: str) -> float | None:
        best: float | None = None
        for cond in rp.NON_ORACLE:
            m = rp._mean_metric(agg, cond, algorithm, rp.HEADLINE_CPDAG_METRIC)
            if m is None:
                continue
            if best is None or m > best:
                best = float(m)
        return best

    print("Interventional headline (CPDAG F1), gold=original:")
    print("  C0 (obs → int):")
    for alg in ALGORITHMS_INTERV:
        o = rp._mean_metric(agg_obs, "C0", alg, rp.HEADLINE_CPDAG_METRIC)
        i = rp._mean_metric(agg_int, "C0", alg, rp.HEADLINE_CPDAG_METRIC)
        if alg == "GIES":
            print(f"    {alg}: N/A → {i:.2f}" if i is not None else f"    {alg}: N/A")
        elif o is None or i is None:
            print(f"    {alg}: N/A")
        else:
            print(f"    {alg}: {o:.2f} → {i:.2f}")
    print("  Best non-oracle (obs → int):")
    for alg in ALGORITHMS_INTERV:
        o = _best_non_oracle(agg_obs, alg)
        i = _best_non_oracle(agg_int, alg)
        if alg == "GIES":
            print(f"    {alg}: N/A → {i:.2f}" if i is not None else f"    {alg}: N/A")
        elif o is None or i is None:
            print(f"    {alg}: N/A")
        else:
            print(f"    {alg}: {o:.2f} → {i:.2f}")
    print("  C5 oracle (obs → int):")
    for alg in ALGORITHMS_INTERV:
        o = rp._mean_metric(agg_obs, "C5", alg, rp.HEADLINE_CPDAG_METRIC)
        i = rp._mean_metric(agg_int, "C5", alg, rp.HEADLINE_CPDAG_METRIC)
        if alg == "GIES":
            print(f"    {alg}: N/A → {i:.2f}" if i is not None else f"    {alg}: N/A")
        elif o is None or i is None:
            print(f"    {alg}: N/A")
        else:
            print(f"    {alg}: {o:.2f} → {i:.2f}")


def main() -> None:
    obs_path = REPO_ROOT / "experiments" / "ablation_results_sachs.json"
    int_path = REPO_ROOT / "experiments" / "ablation_results_sachs_interventional.json"
    obs_full = json.loads(obs_path.read_text(encoding="utf-8"))["results"]
    int_full = json.loads(int_path.read_text(encoding="utf-8"))["results"]

    obs_orig = rp.filter_results_by_gold(obs_full, "original")
    int_orig = rp.filter_results_by_gold(int_full, "original")

    agg_obs = rp.aggregate(obs_orig)
    agg_int = _aggregate_four(_interventional_table_rows(int_orig))

    int_llm = next(
        (
            r
            for r in int_orig
            if r.get("condition") == "C-LLM-only"
            and str(r.get("gold_version") or "original") == "original"
        ),
        None,
    )
    if int_llm is None:
        raise ValueError(
            "interventional ablation JSON missing C-LLM-only row (original gold)"
        )

    tables_dir = REPO_ROOT / "tables"
    write_ablation_table_interventional(
        agg_int, int_llm, tables_dir / "ablation_table_interventional.tex"
    )
    write_observational_vs_interventional(
        agg_obs, agg_int, tables_dir / "observational_vs_interventional.tex"
    )
    _print_interventional_headline(agg_obs, agg_int)
    print(
        "Wrote tables/ablation_table_interventional.tex, "
        "tables/observational_vs_interventional.tex"
    )


if __name__ == "__main__":
    main()
