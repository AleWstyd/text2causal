# Step 7' — Real Public Second Dataset

Step 7' replaces the synthetic DREAM4 fallback with a real, public,
direct-download signalling benchmark: **LiverDREAM / CellNOpt** from
Saez-Rodriguez et al. and the CellNOptR public model zoo.

## Acquisition Decision

The planned Day-1 search was:

1. HPN-DREAM Breast Cancer Network Inference.
2. DREAM4 Predictive Signalling Network Modeling.
3. Saez-Rodriguez / CellNOpt liver-signalling data.

HPN-DREAM and DREAM4 PSN both resolve primarily to Synapse challenge projects in
this environment. Their project metadata is visible, but direct file access is
not available without Synapse-authenticated downloads and challenge file IDs.
The direct public route that works reproducibly is CellNOptR's model zoo.

The chosen dataset is:

- data: `data/liverdream/raw/MD-LiverDREAM.csv`
- prior knowledge network: `data/liverdream/raw/PKN-LiverDREAM.sif.txt`
- source URLs: `https://raw.githubusercontent.com/saezlab/CellNOptR/gh-pages/public/`

## Dataset Construction

The runner `experiments/run_liverdream_real.py` processes the MIDAS-style file
into seven observed readout variables:

- `akt`
- `mek12`
- `erk12`
- `ikb`
- `jnk12`
- `p38`
- `hsp27`

The ground-truth graph is a latent projection of the LiverDREAM SIF prior
knowledge network onto those observed readouts. An observed edge `u -> v` is
emitted when the PKN contains a directed path from `u` to `v` whose internal
nodes are unobserved. The resulting five-edge graph is stored at
`data/liverdream/ground_truth.gml`.

This is not synthetic: the data and PKN are public CellNOpt artefacts. The
ground truth is literature-derived rather than experimentally held out, so it
should be described as a real public signalling benchmark with a curated PKN
reference graph.

## Artefacts

The run emits:

- `data/liverdream/data.csv`
- `data/liverdream/ground_truth.gml`
- `data/liverdream/background.txt`
- `grounding/gold_liverdream.json`
- `experiments/grounding_liverdream.json`
- `experiments/causal_priors_liverdream.json`
- `experiments/floor_priors_liverdream_all.json`
- `experiments/floor_priors_liverdream_reactome_only.json`
- `experiments/freetext_priors_liverdream.json`
- `experiments/oracle_priors_liverdream.json`
- `experiments/ablation_results_liverdream.json`
- `experiments/constraint_quality_liverdream.json`
- `tables/ablation_table_liverdream.tex`
- `tables/cross_dataset.tex`

## Results

The real-dataset acceptance gate passes on algorithmic improvement:

- PC improves from C0 F1 = 0.29 to C2/C3/C4 F1 = 0.33.
- The improvement is +0.04 F1, clearing the +0.03 gate.
- GES remains flat at F1 = 0.44 for C0/C2/C3/C4.
- LiNGAM remains flat at F1 = 0.46 for C0/C2/C3/C4.
- The C5 oracle ceiling is reachable: PC F1 = 0.44, GES F1 = 0.83,
  LiNGAM F1 = 0.67.

Constraint quality on LiverDREAM is conservative:

- Reactome+LLM emits one forward prior at threshold 0.7.
- That prior is correct (`mek12 -> erk12`): precision = 1.00.
- Recall is low: 0.20.
- Strict hallucination is 0.00.

This strengthens the project because the same qualitative pattern from Sachs
appears on a real second dataset: Reactome+LLM priors are low-hallucination and
can produce a modest CD improvement, but coverage remains the limiting factor.

## Verification

- `task liverdream`

The LiverDREAM ablation completed 211/211 cells with 0 failures.
