# Step 7 — DREAM4 Predictive Signalling Integration

**Role.** Evaluation. Demonstrates the generalisation claim with a second dataset.

**Goal.** Apply the full pipeline to the DREAM4 Predictive Signalling Network Modelling dataset (MCF7 breast cancer, ~7 antibody-measured human signalling proteins) with **no code changes** — only configuration changes.

**Depends on.** Steps 1–6 complete and clean on Sachs.

**Effort.** 2–3 days.

---

## Deliverables

| Artefact | Purpose |
|----------|---------|
| `data/dream4_psn/data.csv` | Observational data |
| `data/dream4_psn/ground_truth.gml` | Ground-truth network |
| `data/dream4_psn/background.txt` | Free-text background paragraph for C1 |
| `grounding/gold_dream4_psn.json` | Hand-curated gold-standard grounding |
| `experiments/grounding_dream4_psn.json` | LLM-produced grounding |
| `experiments/causal_priors_dream4_psn.json` | LLM-derived priors |
| `experiments/floor_priors_dream4_psn_*.json` | OmniPath floors |
| `experiments/freetext_priors_dream4_psn.json` | C1 priors |
| `experiments/oracle_priors_dream4_psn.json` | C5 priors |
| `experiments/ablation_results_dream4_psn.json` | Per-cell metrics |
| `experiments/constraint_quality_dream4_psn.json` | Constraint quality |
| `tables/ablation_table_dream4.tex`, `tables/cross_dataset.tex` | Cross-dataset comparison |
| `figures/gap_closed_dream4.{pdf,png}` | Per-dataset and cross-dataset figures |

---

## Implementation tasks

### Day 1 (morning) — Data acquisition + tripwire

1. **Acquire DREAM4 PSN data.** Try in this order:
   - Synapse (sage bionetworks): the official DREAM challenge archive.
   - The supplementary materials of Prill et al. (PLoS ONE 2010) or Saez-Rodriguez et al. (Mol. Syst. Biol. 2009) — DREAM4 PSN is described there.
   - GitHub mirrors of the DREAM4 challenge data.

2. **Verify accessibility on Day 1.** If the data is paywalled, behind a defunct Synapse account, or otherwise inaccessible:
   - **Tripwire fired.** Substitute synthetic SCM data sampled from a Reactome-derived human pathway DAG.
   - Recommended substitute: extract a 7–10 node DAG from the Reactome PI3K/Akt or RAS/MAPK pathway, sample data from a linear non-Gaussian SCM (matches LiNGAM's assumptions), and use that as the secondary dataset.
   - Document the substitution clearly in `docs/dev_plan_v2.md` §3 Step 7 (update the changelog) and in the paper.

3. **Format data.** Convert to CSV (continuous) under `data/dream4_psn/data.csv`. Format ground truth as a directed `networkx` graph and persist as GML.

### Day 1 (afternoon) — Coverage check

4. **Run Reactome coverage check** for DREAM4 PSN nodes (use the `experiments/reactome_coverage.py` from Step 2).
   - Expected outcome: high coverage. DREAM4 PSN proteins (AKT, ERK1, MEK, p70S6K, etc.) are mainstream Reactome territory.
   - If `node_coverage < 0.8`, fall back to OmniPath as primary structured source for DREAM4 PSN. Do NOT abandon DREAM4 PSN over this — OmniPath aggregates Reactome plus 30+ other sources, so a low Reactome score doesn't preclude useful priors.

### Day 2 — Gold standard + free-text background

5. **Hand-curate `grounding/gold_dream4_psn.json`.** Should be ~7 entries; mostly proteins, possibly one or two complexes (PI3K is a heterodimer). Verify each UniProt ID against the source paper.

6. **Write `data/dream4_psn/background.txt`.** 1–2 paragraphs, textbook-level description of MCF7 signalling biology. Same constraint as Sachs: do NOT leak ground-truth edges. Used for the C1 free-text-LLM condition.

### Day 2 (afternoon) — Run pipeline

7. **Run the full pipeline on DREAM4 PSN.** No code changes — only:
   ```
   task dream4-grounding   # Step 3 with --dataset dream4_psn
   task dream4-priors      # Step 4 with --dataset dream4_psn
   task ablation -- --dataset dream4_psn
   ```
   The runner from Step 6 should be parameterised on `dataset_name`. If it isn't, fix that — it's a configuration issue, not a code change to the scientific pipeline.

8. **If the runner has any dataset-specific code paths**, that's a bug. The whole point of this step is to demonstrate generalisation. Refactor those code paths into config.

### Day 3 — Cross-dataset reporting

9. **Extend `experiments/report.py`** to produce cross-dataset tables:
   - `tables/cross_dataset.tex`: side-by-side mean ± std for Sachs and DREAM4 PSN, per condition × algorithm × metric.
   - `figures/gap_closed_dream4.pdf`: per-dataset gap-closed figure.
   - A combined figure showing whether the same condition (e.g. C3) is best on both datasets.

10. **Compute the generalisation claim's empirical support.**
    - Does the Reactome+LLM pipeline beat C0 on both datasets?
    - Does the same threshold (τ = 0.7) work well on both?
    - Are the headline numbers (% of C0→C5 gap closed) comparable, or does one dataset benefit much more?
    - Document the answer in the discussion section's notes — this informs Step 8's writing.

---

## Acceptance criteria

- [ ] DREAM4 PSN data is loaded, ground truth is loaded, background paragraph exists.
- [ ] Reactome (or OmniPath, if tripwire fired) coverage on DREAM4 PSN nodes ≥ 0.8.
- [ ] Grounding accuracy on DREAM4 PSN ≥ 0.9 vs gold standard.
- [ ] Full ablation runs cleanly on DREAM4 PSN with no code modifications — only config.
- [ ] Cross-dataset table and figures produced.
- [ ] `task lint` and `task test` pass.

---

## Pitfalls

- **DREAM4 PSN data accessibility is the single biggest risk.** Verify on Day 1 morning. If unavailable, do not waste time chasing — fall back to synthetic-from-Reactome immediately.
- **Synthetic substitute is an honest fallback, not a downgrade.** A Reactome-derived DAG with sampled data is methodologically *cleaner* than DREAM4 PSN in some ways (no measurement noise, exact ground truth). Frame it correctly in the paper: "secondary evaluation on a synthetic dataset derived from Reactome" — this is a valid generalisation argument, just a different one.
- **Configuration leakage.** The Step 6 runner should accept `--dataset` as a CLI flag and read paths from a config file (or env). If it has hard-coded `'sachs'` strings anywhere, that's a bug.
- **Gold standard mismatch.** DREAM4 PSN measures phosphorylated states using specific antibodies; the column names may use vendor / clone abbreviations (e.g. `p_AKT_S473`). Verify against the source paper before curating.
- **Smaller graph = noisier metrics.** With 7 nodes and ~10 ground-truth edges, single-edge errors swing F1 by ~10 percentage points. Report median and IQR alongside mean ± std.

---

## Out of scope

- DREAM4 Network Inference (gene-regulatory subset). Not Reactome-coverable; see `dev_plan_v2.md` §1.4 for the rationale.
- Cross-organism evaluation (E. coli, yeast).
- Paper writing (Step 8).
