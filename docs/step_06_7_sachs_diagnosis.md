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

## fGES Decision

The optional `pytetrad` fGES route was checked and skipped for this
strengthening pass. `pytetrad` is not available from the package registry in
this environment (`uv pip install --dry-run pytetrad` fails with "not found").
The available route is the CMU GitHub package (`cmu-phil/py-tetrad`), which
brings a JPype/Java bridge and would be a dependency-integration task rather
than a small Step 6.7 diagnostic.

The GES post-hoc constraint caveat therefore remains.

## Verification

- `task ablation-sachs`
- `task report-sachs`
- `uv run python -m experiments.per_edge_attribution_sachs`

The canonical Sachs ablation has 241/241 cells, all successful (the
previously N/A C5 oracle LiNGAM seeds now complete via post-hoc
required-edge injection; see PR3b).
