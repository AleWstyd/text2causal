# Step 8 — Interventional Sachs + GIES (PR6)

## Scope

Add the published Sachs *et al.* (2005) **nine experimental contexts** as first-class data, run a **parallel** Step 6-style ablation on the pooled interventional matrix, and introduce **GIES** (Hauser & Bühlmann, *JMLR* 2012) as an interventional-native score-based learner. The legacy **observational** pipeline (`load_sachs_dataset`, `task ablation-sachs`) is unchanged.

## Dataset

- **Source:** Zenodo repackaging DOI [10.5281/zenodo.7681811](https://doi.org/10.5281/zenodo.7681811) of the Science supplementary flow-cytometry files (nine real conditions; simulated `cd3cd28icam2_*` variants are excluded). Primary citation: Sachs *et al.*, *Science* **308**, 523–529, 2005 ([DOI 10.1126/science.1105809](https://doi.org/10.1126/science.1105809)).
- **Layout:** `data/sachs/interventional/raw/*.csv` + `manifest.json` + `README.md` (citations + licence notes).
- **Alignment with CDT:** CSV columns are renamed on load to match `cdt.data.load_dataset("sachs")`. The nine files sum to **7466** rows — the same pooled matrix as the committed observational Sachs bundle — so **PC / GES / LiNGAM “naive” cells reproduce observational scores** when priors and seeds match; the contrast is methodological (same numbers confirm no accidental data skew) while **GIES** exploits environment structure.

## Algorithm

- **Library:** [`gies`](https://pypi.org/project/gies/) v0.0.3 (Gamella & Kolotuhina; BSD 3-clause), a pure NumPy port of GIES validated against `pcalg`. **R/CDT `pcalg::gies` was not used** (`Rscript` unavailable here; `cdt.causality.graph.GIES` remains R-backed).
- **Declined alternatives:** Python `bnlearn`’s `import_example("sachs")` simulates from a discrete BIF (not the real interventional cytometry). `gcastle` wheel exposes GES but not GIES. `causaldag` failed import (XGBoost **OpenMP** / `libomp` missing on this macOS image).

## Runner & metrics

- **Script:** `experiments/runner_interventional.py` → `experiments/ablation_results_sachs_interventional.json` (802 cells: 400 algorithmic × 2 golds + 2 C-LLM-only).
- **Algorithms:** `PC`, `GES`, `LiNGAM` with `intervention_strategy="naive"` (ignore manifest targets); `GIES` runs both **naive** (single pooled environment) and **gies** (nine environments + manifest `I`).
- **Constraints:** Same post-hoc edit pass as GES (`apply_post_hoc_edits`). Cycle drops are logged per algorithm; GIES oracle runs can accumulate many post-hoc drops because required edges attach to a dense CPDAG output.

## Reporting

- `task report-sachs-interventional` → `tables/ablation_table_interventional.tex`, `tables/observational_vs_interventional.tex`.
- **Headline (CPDAG F1, gold=original, May 2026 run):**
  - Best non-oracle GIES (`gies`): **0.53** vs PC/GES/LiNGAM best **0.51 / 0.24 / 0.45** (observational and interventional naive agree for the latter three).
  - C5 oracle GIES (`gies`): **0.71** vs PC **0.51**, GES **0.24**, LiNGAM **0.48**.
  - C0 GIES (`gies`): **0.20** (strong global score penalty at τ=0 without priors is expected).

## Caveats

1. **Manifest targets** map each drug condition to a primary column index for `gies.fit_bic`; pharmacology is simplified (e.g. PI3K inhibition → `PIP3` node). Sensitivity to this mapping should be discussed in prose.
2. **Identical naive scores** across observational vs interventional JSONs for PC/GES/LiNGAM are a **sanity check**, not a negative result — the pooled matrix matches CDT.
3. **Post-hoc constraints** on GIES outputs share the same structural caveat as GES (Risk #6 in the dev plan).

## Acceptance

- [x] `task test` passes offline (`HTTPS_PROXY=http://127.0.0.1:1`).
- [x] `task ablation-sachs-interventional` is independent of `task ablation`.
- [x] No changes to `load_sachs_dataset()` defaults.
- [x] Documented data lineage + algorithm rationale in this spec and `docs/dev_plan_v2.md` changelog item 23.
