# Step 6 — Ablation Study + Constraint Quality Eval

**Role.** Evaluation. Produces the headline numbers for the paper.

**Goal.** Run the full ablation across all conditions × algorithms × seeds on Sachs; evaluate constraint quality independently of discovery; produce paper-ready tables and figures.

**Depends on.** Steps 1–5.

**Effort.** 3–4 days.

---

## Deliverables

| Artefact | Purpose |
|----------|---------|
| `experiments/runner.py` | Orchestrates the full sweep, idempotent |
| `experiments/ablation_results_sachs.json` | Per-cell metrics, mean ± std aggregations |
| `experiments/constraint_quality_sachs.json` | Precision/recall/hallucination/coverage of priors |
| `experiments/cost_report.json` | Tokens, dollars, wall-clock per stage |
| `tables/ablation_table.tex` | LaTeX table per metric × condition × algorithm (**skeleton F1** — undirected overlap; see caption) |
| `tables/ablation_table_directed.tex` | Same layout with **directed F1** (ordered arcs); headline table for orientation-aware reporting |
| `tables/aupr_extension.tex` | PC-only AUPR + skeleton F1 + directed F1 (`F1\textsubscript{dir}`) |
| `tables/constraint_quality.tex` | Constraint-quality LaTeX table |
| `figures/gap_closed.{pdf,png}` | Headline figure: % of C0→C5 gap closed per condition × algorithm; includes a separate C-LLM-only bar (no algorithm dimension) |
| `figures/cd_vs_llm_only.{pdf,png}` | Direct comparison: best LLM+CD condition (C2/C3) vs C-LLM-only, per dataset |
| `figures/threshold_sensitivity.{pdf,png}` | Metric vs τ ∈ {0.6, 0.7, 0.8, 0.9} |
| `task ablation` Taskfile target | Single-command regeneration |

---

## Ablation conditions (Sachs)

| ID | Condition | Priors source | Threshold | Algorithm dim? |
|----|-----------|---------------|-----------|----------------|
| C0 | No knowledge | — | — | yes (PC/GES/LiNGAM × 10 seeds) |
| C0.5 | OmniPath direct edges, no LLM | `omnipath_floor_reactome_only` | 0.7 | yes |
| C1 | Free-text-LLM (current skeleton on Sachs background paragraph) | `freetext_llm` | 0.7 | yes |
| C2 | Reactome+LLM, hard only | `reactome_llm` | 0.9 | yes |
| C3 | Reactome+LLM, hard + soft | `reactome_llm` | 0.7 | yes |
| C4 | Reactome+LLM, hard + soft (lower threshold) | `reactome_llm` | 0.6 | yes |
| **C-LLM-only** | **LLM judgments emitted as DAG directly (from `predicted_dag_llm_only_sachs.gml`)** | `reactome_llm` | 0.7 | **no — single deterministic DAG** |
| C5 | Oracle | `ground_truth` | — | yes |

Cell count per dataset:
- Conditions with algorithm dimension (C0, C0.5, C1, C2, C3, C4, C5): 7 × 3 algorithms × 10 seeds = **210 cells**.
- C-LLM-only: 1 deterministic DAG = **1 cell** (no algorithm, no seed).
- **Total ≈ 211 cells per dataset.**

---

## Implementation tasks

### Sweep runner

1. **Implement `experiments/runner.py`.**
   ```python
   def run_full_ablation(
       dataset_name: str,
       conditions: list[Condition],
       algorithms: list[str],
       seeds: list[int],
       output_path: Path,
   ) -> None: ...
   ```
   - Load dataset, ground truth, all priors files (LLM, floor, free-text, oracle).
   - For each `(condition, algorithm, seed)` cell, call `experiments.run_condition.run_condition`.
   - Append result to JSON file. Idempotent — skip already-completed cells.
   - Print progress (cell N/M, ETA).

2. **Generate the C1 (free-text-LLM) priors.**
   - Reuse the existing `llm/extract_relations.py` extractor.
   - Use the Sachs background paragraph (write a short one to `data/sachs/background.txt` if it doesn't exist — keep it minimal, e.g. a 2-paragraph description of Sachs et al. 2005 biology that does NOT leak ground-truth edges).
   - Persist to `experiments/freetext_priors_sachs.json` in the same schema as `causal_priors_sachs.json`.

3. **Generate the C5 oracle priors.**
   - Read ground-truth DAG from `cdt.data.load_dataset('sachs')`.
   - For each ground-truth edge `(a, b)`, emit `{cause: a, effect: b, confidence: 1.0, constraint_type: "hard_required"}`.
   - Persist to `experiments/oracle_priors_sachs.json`.

4. **Wire the C-LLM-only condition into the runner.**
   - Load `experiments/predicted_dag_llm_only_sachs.gml` (produced in Step 4).
   - Evaluate it directly against ground truth via `evaluation/harness.py`.
   - Store one row in `ablation_results_sachs.json` with `condition: "C-LLM-only"`, `algorithm: null`, `seed: null`, the metrics, and the predicted edges.
   - No algorithm runs; no seed loop; no statistical fitting. This is the cheapest cell in the entire sweep.

### Constraint quality evaluator

5. **Implement `experiments/constraint_quality.py`.**
   - For each priors source (`reactome_llm`, `omnipath_floor`, `freetext_llm`):
     - Filter at τ = 0.7.
     - Treat as a predicted directed edge set.
     - Compute against ground truth:
       - **Precision** = `|predicted ∩ true| / |predicted|`.
       - **Recall** = `|predicted ∩ true| / |true|`.
       - **Hallucination rate** = fraction of `predicted` edges where the *reverse* edge is in `true` (clear-cut wrong-direction calls).
       - **Coverage rate** = `|pairs_with_context| / |all_pairs|`. For OmniPath: `|pairs_with_omnipath_edges| / |all_pairs|`.
   - Write `experiments/constraint_quality_sachs.json`:
     ```json
     {
       "reactome_llm": {"precision": 0.78, "recall": 0.41, "hallucination": 0.06, "coverage": 0.79},
       "omnipath_floor": {"precision": 0.62, "recall": 0.53, "hallucination": 0.04, "coverage": 0.85},
       "freetext_llm": {"precision": 0.71, "recall": 0.29, "hallucination": 0.12, "coverage": 1.00}
     }
     ```

### Reporting

6. **Implement `experiments/report.py`.**
   - Loads `ablation_results_sachs.json`, computes mean ± std per `(condition, algorithm, metric)` for the algorithmic conditions; reports point estimates for C-LLM-only.
   - Generates `tables/ablation_table.tex` (LaTeX `tabular` with mean ± std). C-LLM-only is reported in a separate row that spans the algorithm columns (single value, no per-algorithm dimension), with a footnote explaining it is algorithm-free.
   - Generates `tables/constraint_quality.tex` from `constraint_quality_sachs.json`.
   - Generates `figures/gap_closed.pdf`:
     - X-axis: algorithm (PC, GES, LiNGAM).
     - Y-axis: % of C0→C5 gap closed = `(F1[Cx] - F1[C0]) / (F1[C5] - F1[C0])`.
     - One bar group per condition (C0.5, C1, C2, C3, C4).
     - Add a horizontal dashed line at the C-LLM-only F1 (or C-LLM-only gap-closed value) for direct visual comparison.
   - Generates `figures/cd_vs_llm_only.pdf`:
     - X-axis: dataset (Sachs, DREAM4 PSN once Step 7 done).
     - Y-axis: F1.
     - Three bars per dataset: best LLM+CD condition (max over C2/C3 × algorithm), C-LLM-only, oracle (C5 best). Direct visual answer to "does CD add value?"
   - Generates `figures/threshold_sensitivity.pdf`:
     - X-axis: τ ∈ {0.6, 0.7, 0.8, 0.9}.
     - Y-axis: F1.
     - One line per algorithm.
   - Identifies and prints (a) the best condition per algorithm and the absolute improvement vs C0, and (b) the **CD-vs-LLM-only delta**: `F1[best LLM+CD] - F1[C-LLM-only]` per dataset, with sign and magnitude.

### Cost / latency report

7. **Implement `experiments/cost_report.py`.**
   - Walk the LLM cache (`cache/llm/**/*.json`).
   - Aggregate per stage tag (grounding / reasoning / freetext): total tokens (prompt + completion), estimated dollars at the current model's published price, wall-clock time stored in cache metadata.
   - Write `experiments/cost_report.json`. Include in paper appendix.
   - Note: this requires Steps 3 and 4 to write a stage tag into cache metadata. Cheap addition.

### Taskfile

8. **Add `task ablation`** that runs:
   ```
   uv run python -m experiments.runner sachs
   uv run python -m experiments.constraint_quality sachs
   uv run python -m experiments.cost_report
   uv run python -m experiments.report sachs
   ```
   This is the single-command reproduction target.

---

## Acceptance criteria

- [ ] All 211 cells complete on Sachs (210 algorithmic + 1 C-LLM-only), or all minus a small documented set of expected failures (e.g. PC under conflict-heavy C4 may legitimately fail).
- [ ] C-LLM-only DAG from Step 4 is loaded, evaluated, and recorded as a single ablation row.
- [ ] Re-running `task ablation` is idempotent — skips completed cells.
- [ ] `tables/ablation_table.tex` and `tables/constraint_quality.tex` compile in LaTeX; ablation table includes the C-LLM-only row.
- [ ] `figures/gap_closed.pdf`, `figures/cd_vs_llm_only.pdf`, and `figures/threshold_sensitivity.pdf` are publication-quality (readable axis labels, legible legend, no clipped text).
- [ ] `experiments/cost_report.json` reports total tokens, dollars, wall-clock per stage.
- [ ] Headline number (% of C0→C5 gap closed by best Reactome+LLM condition) is computed and printed.
- [ ] CD-vs-LLM-only delta is computed and printed per dataset.
- [ ] `task lint` and `task test` pass.

---

## Pitfalls

- **C1 free-text background paragraph must not leak ground-truth edges.** A reviewer will check this. Write the paragraph from a textbook-level description of Sachs's biology, not from the Sachs paper's own causal claims. Keep it under ~200 words.
- **Variance reporting.** Some algorithms (PC) are deterministic given fixed data — `std` will be 0. That's fine but flag it in the table caption.
- **Hallucination rate definition.** "Reverse-of-true" is the strict definition. There's also "edge between two unrelated nodes" — log both, report the strict one in the headline.
- **Threshold ≥ vs >.** Use `≥` consistently with Step 5.
- **Cost report accuracy.** Token counts in cached responses are authoritative. Pricing changes; lock the per-token rate at paper-write time and document the snapshot date.
- **Idempotency edge case.** If a cell crashed mid-write, the JSON file may be corrupted. The runner should atomic-write (write to `.tmp`, rename) per cell.
- **C-LLM-only is not "free" methodologically — only computationally.** It depends on the same cached priors as C2/C3, so the comparison is not independent. State explicitly in the paper that C-LLM-only and C2/C3 share an upstream LLM judgment source; the comparison isolates the *use* of those judgments (constraints vs DAG), not different judgments.
- **C-LLM-only on smaller graphs may dominate trivially.** With 11 Sachs nodes and 17 ground-truth edges, an LLM that simply recalls textbook signalling gets a high score. Report alongside Kıcıman et al. (2023)'s reported numbers on Sachs as a sanity check, not as a competitive contribution.

---

## Out of scope

- DREAM4 (Step 7 — runner is parameterised on `dataset_name`, just call it again with `dream4_psn` once data is integrated).
- Paper writing (Step 8).
- pytetrad fGES (parking lot).
