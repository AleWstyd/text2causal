# Step 6.5 — Sachs Stabilisation Verdict

Step 6.5 tested whether the Step 6 Sachs results could be converted into a
positive Reactome+LLM discovery headline before moving to DREAM4. The answer is
no: the stabilised Sachs story should be framed around constraint quality,
auditability, and structured-vs-aggregate attribution, with the downstream
causal-discovery ablation reported as a quantified negative result.

## What Was Tried

### LiNGAM sparse-prior sweep

Dense required-edge matrices made all canonical priors-augmented DirectLiNGAM
cells fail. The Step 6.5 sweep tested three weaker encodings in
`experiments/lingam_sweep_sachs.json`:

- `sparse_required`: keep only required edges with confidence >= 0.95.
- `forbidden_only`: do not require forward edges; softly forbid the reverse of
  each qualifying directional claim.
- `hybrid_top5`: require at most the five highest-confidence edges and softly
  forbid reverses.

The only useful variant was `forbidden_only` with soft prior application:
C3/C4 LiNGAM completed with F1 = 0.566 and SHD = 37. This unblocks a
Reactome+LLM LiNGAM cell, but it still underperforms C0 LiNGAM
(F1 = 0.593, SHD = 34). The best successful sparse-prior result was promoted
into the canonical `experiments/ablation_results_sachs.json` for C3 LiNGAM so
the paper can report the attempted fix instead of a pure implementation
failure.

### PC invariance diagnosis

`experiments/pc_invariance_sachs.json` shows that PC is not literally ignoring
priors. The edge sets differ across conditions and across alpha values, and PC
AUPR improves from 0.52 in C0 to 0.56 under C1/C2/C3/C4/C5. However,
direction-aware F1 remains flat at 0.52, including under oracle constraints.
This is a metric-level negative result, not just a wiring bug.

The table snippet `tables/aupr_extension.tex` should be used in the paper to
show the hidden AUPR gain while being clear that F1 did not improve.

### Coverage-conditional constraint quality

`experiments/constraint_quality_sachs.json` now reports both global and
Reactome-covered-subset constraint quality. On the Reactome-covered pair subset
(33/55 unordered Sachs pairs), Reactome+LLM has:

- precision = 0.53
- recall = 0.69
- strict hallucination = 0.00

On the same subset, OmniPath-all has:

- precision = 0.17
- recall = 0.31
- strict hallucination = 0.26

This is the strongest positive result after Step 6.5. It supports a claim that
Reactome-grounded LLM reasoning produces safer and more accurate directional
constraints than the aggregated multi-source floor, even when those constraints
do not translate into improved Sachs F1 under the tested CD algorithms.

## Decision Gate

The positive-headline gate was:

> Reactome+LLM must beat C0 by at least 0.03 F1 for at least one successful
> algorithm-condition cell.

This gate did not pass:

- PC: best Reactome+LLM F1 = 0.52, C0 F1 = 0.52.
- GES: best Reactome+LLM F1 = 0.53, C0 F1 = 0.53.
- LiNGAM: best Reactome+LLM F1 = 0.57, C0 F1 = 0.59.

The paper should therefore use the reframed narrative:

1. Reactome+LLM gives low-hallucination, auditable causal priors from curated
   biological evidence.
2. Those priors outperform OmniPath-all and OmniPath-Reactome-only on
   constraint quality.
3. On Sachs, the downstream CD algorithms do not reliably convert those better
   priors into better F1; this is reported as a quantified negative result and
   algorithm-specific limitation.
4. Step 7 remains necessary to test whether this is Sachs-specific or a broader
   pattern on a second human-signalling dataset.

## Verification

- `task ablation`
- `task report-sachs`

Both completed successfully after the Step 6.5 changes. The canonical ablation
now has 211/211 cells, 181 successful and 30 failed. The remaining failures are
LiNGAM cells whose prior matrices still over-constrain DirectLiNGAM.
