# Step 6.7 — Sachs CD Diagnosis

Step 6.7 turns the Sachs downstream-discovery result from an unresolved null
result into a diagnosed one. The headline remains unchanged: Reactome+LLM
improves prior quality, but the tested causal-discovery algorithms do not
reliably convert those priors into a Sachs F1 improvement.

## What Changed

### Canonical Reactome+LLM LiNGAM cells now complete

Step 6.5 showed that dense required-edge matrices over-constrain
`DirectLiNGAM`. The canonical Sachs runner now uses the successful
`forbidden_only` sparse LiNGAM encoding for Reactome+LLM conditions C2/C3/C4:

- C2 LiNGAM: F1 = 0.56, SHD = 38.
- C3 LiNGAM: F1 = 0.57, SHD = 37.
- C4 LiNGAM: F1 = 0.57, SHD = 37.

This removes the Reactome+LLM LiNGAM N/A cells from
`tables/ablation_table.tex`. The C5 oracle LiNGAM cell is still N/A because it
keeps the true oracle semantics: true edges are required, not merely used to
forbid reverse directions. Weakening C5 would make the row complete but no
longer an oracle ceiling.

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
4. LiNGAM is the limiting algorithm. Sparse forbidden-only priors complete
   successfully, but they still underperform C0 LiNGAM and worsen 3/5
   reaction-stratum true edges.

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

The canonical Sachs ablation has 211/211 cells, with 201 successful and 10
failed. The remaining failures are the dense-required C5 oracle LiNGAM seeds.
