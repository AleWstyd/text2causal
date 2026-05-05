# Step 6.7 — Sachs CD Diagnosis

Step 6.7 turns the Sachs downstream-discovery result from an unresolved null
result into a diagnosed one. The headline remains unchanged: Reactome+LLM
improves prior quality, but the tested causal-discovery algorithms do not
reliably convert those priors into a Sachs F1 improvement.

## What Changed

### Per-pair fallback for the no-context stratum (PR2b)

Paragraph `freetext_priors_sachs.json` does not intersect `no_context_pairs`, so the first **C3+ft** merge (`n_fallback_applied = 0`) left Sachs identical to **C3**. PR2b adds `task per-pair-freetext-fallback`: one cached LLM JSON per *unordered* Reactome no-context pair (`reasoning/freetext_fallback.py`), merged in a second round by `task fallback-priors`. On the committed May 2026 artefact run: **24** Sachs queries, **18** accepted directional claims, **36** prior-slot substitutions when both ordered rows exist; downstream **GES** mean directed F1 moves **0.491 → 0.546** (SHD **27 → 25**) and **LiNGAM** **0.400 → 0.453** (**33 → 29**); **PC** is unchanged (**0.524**, SHD **20**). All **6** true Sachs edges in the no-context stratum (PKA/PKC substrate arcs) are oriented correctly in `per_pair_freetext_priors_sachs.json`.

### Canonical Reactome+LLM LiNGAM cells now complete

Step 6.5 showed that dense required-edge matrices over-constrain
`DirectLiNGAM`. Step 6.7 adopted the `forbidden_only` soft sparse encoding to
make the Reactome+LLM LiNGAM cells succeed. Step 6.7b (May 2026, PR3b) retires
that encoding for the canonical Sachs matrix because the soft-prior path could
not actually block reverse edges — DirectLiNGAM's 0.001 coefficient threshold
let wrong-direction arcs survive, so C5 oracle LiNGAM returned directed F1=0
even though the priors are perfect. The canonical Reactome+LLM LiNGAM cells
now run **unconstrained** and apply the same `apply_post_hoc_edits`
required-add / forbidden-remove pass as GES and PC
(`lingam_prior_mode="post_hoc"`):

- C2 LiNGAM: skeleton F1 = 0.50, directed F1 = 0.36, SHD = 36.
- C3 LiNGAM: skeleton F1 = 0.44, directed F1 = 0.40, SHD = 33.
- C4 LiNGAM: skeleton F1 = 0.44, directed F1 = 0.40, SHD = 33.
- C5 LiNGAM: skeleton F1 = 0.48, directed F1 = 0.48, SHD = 26.

The C5 oracle LiNGAM cell is now populated with a real post-hoc
required-edge injection result; the 7 oracle-required edges that close a
cycle against LiNGAM's unconstrained prediction are dropped and logged as
`dropped_due_to_cycle` (no more N/A). The three native-LiNGAM encodings
(`all`, `sparse_required`, `forbidden_only`, `hybrid_top5`) stay selectable
via `lingam_prior_mode` for the standalone sweep in
`experiments/lingam_sweep_sachs.py` and remain documented in Step 6.5.

### Reaction-stratum per-edge attribution added

`experiments/per_edge_attribution_sachs.json` diagnoses what happens on the five
Sachs true edges with direct Reactome reaction-level evidence. Comparing C3
against C0:

- PC improves one of five reaction-stratum edge orientations and leaves four
  unchanged.
- GES improves two of five and leaves three unchanged.
- LiNGAM worsens three of five and leaves two unchanged.

This explains why the strong reaction-stratum prior-quality number
(precision = 0.83, recall = 1.00) does not become a downstream LiNGAM F1 win.
The priors help PC/GES orientation locally, but LiNGAM's ordering search reacts
poorly even to the weaker forbidden-only encoding.

## Updated Sachs Interpretation

The Sachs result should be interpreted as:

1. Reactome+LLM produces safer directional priors than OmniPath-all on covered
   Sachs pairs.
2. Direct reaction-level evidence is especially strong: precision = 0.83,
   recall = 1.00, strict hallucination = 0.00.
3. PC and GES show local reaction-stratum orientation gains, and PC still shows
   an AUPR gain (0.52 to 0.56). **Skeleton** F1 (undirected edge overlap) stays
   flat across PC conditions; **directed** F1 is the headline orientation metric
   in Step 6 reporting (`tables/ablation_table_directed.tex`) and can move when
   recovered skeletons match but arc directions differ.
4. LiNGAM post-hoc constraint injection (PR3b) replaces the PR3 soft
   `forbidden_only` encoding so the oracle actually fires (C5 directed
   F1 goes 0.00 → 0.48, SHD 46 → 26, with 7 cycle-closing oracle edges
   logged as dropped). Directed F1 now reads C0 0.37 < C3 0.40 < C5 0.48,
   a monotonic dose-response curve. Skeleton F1 drops (C0 0.59 → C3 0.44)
   because post-hoc strips wrong-direction duplicates that happened to
   count as "skeleton hits" — the drop is by design, not a regression.

## PR4 update (CPDAG metrics + GES adjacency fix)

PR4 adds `cpdag_f1` / `shd_cpdag` and exports PC/GES CPDAG undirected edges as
mutual `(u,v)+(v,u)` arcs in the `nx.DiGraph` passed to evaluation.

`run_ges` previously read only `adj[i,j]==1`, which does not match the
`causal-learn` GES documentation (`G[j,i]=1` and `G[i,j]=-1` for `i → j`, and
`-1` on both endpoints for undirected). The corrected converter aligns with PC
and with the library docstring; **Sachs ablation JSON was regenerated** so GES
rows are not comparable to pre-PR4 committed numbers.

**Sachs headline (mean over seeds; regenerated artefact):**

- **PC:** C0 CPDAG F1 = **0.47** → best non-oracle **C1** CPDAG F1 = **0.51**
  (directed: **0.45** → **0.50**). Oracle **C5:** CPDAG F1 **0.51**, directed **0.50**.
- **GES:** C0 CPDAG F1 = **0.18** → best non-oracle **C3+ft** CPDAG F1 = **0.24**
  (directed: **0.19** → **0.23**). Oracle **C5:** CPDAG F1 **0.24**, directed **0.23**.
- **LiNGAM:** `cpdag_*` equals `directed_*` (fully directed output).

**PC oracle C5:** still **7** `dropped_due_to_cycle` edges per seed
(**70** summed over 10 seeds). Orienting undirected pairs in
`apply_post_hoc_edits` does not reduce this on the current Sachs oracle run
(the failing required injections are not resolved by removing a single reverse
arc alone).

Primary Step 6 table for orientation scoring: **`tables/ablation_table_cpdag.tex`**
(CPDAG F1 headline); **`tables/ablation_table_directed.tex`** remains for
strict directed overlap.

## fGES Decision

The optional `pytetrad` fGES route was checked and skipped for this
strengthening pass. `pytetrad` is not available from the package registry in
this environment (`uv pip install --dry-run pytetrad` fails with "not found").
The available route is the CMU GitHub package (`cmu-phil/py-tetrad`), which
brings a JPype/Java bridge and would be a dependency-integration task rather
than a small Step 6.7 diagnostic.

The GES post-hoc constraint caveat therefore remains.

## PR5 update (dual gold: Mooij et al. 2020)

Reporting now scores the **same** predicted graphs against **`original`** (CDT consensus) and **`mooij2020`** (`data/sachs/gold_mooij2020.gml`). On the May 2026 committed sweep, **best non-oracle CPDAG F1** is **PC 0.51 vs 0.45**, **GES 0.24 vs 0.16**, **LiNGAM 0.45 vs 0.37** (robust headline minima: 0.45 / 0.16 / 0.37). Mooij gold removes/redirects several consensus arcs (Raf–Mek, PKC–P38) and adds Akt→Erk, so metrics move in both directions depending on the predictor; the paper should lead with the robust min line. Extra tables: `tables/ablation_table_cpdag_mooij.tex`, `tables/gold_comparison.tex`.

## Verification

- `task ablation-sachs`
- `task report-sachs`
- `uv run python -m experiments.per_edge_attribution_sachs`

The canonical Sachs ablation has 482/482 cells (240 algorithmic conditions × 2 golds + 2 C-LLM-only), all successful (the
previously N/A C5 oracle LiNGAM seeds now complete via post-hoc
required-edge injection; see PR3b).
