# text2causal

`text2causal` is a Python 3.12 research pipeline for deriving auditable causal
priors for protein-signalling datasets from Reactome evidence and LLM reasoning,
then evaluating those priors with PC, GES, and DirectLiNGAM.

The current paper framing is based on the Step 6.5 stabilisation: Reactome+LLM
constraints are low-hallucination and auditable, but on Sachs the tested causal
discovery algorithms do not reliably convert that constraint quality into better
direction-aware F1.

## Quickstart

```bash
task install
task ablation
task dream4-psn
task paper
```

The committed `cache/llm/` and `cache/reactome/` artefacts let Sachs results
replay without API credentials. A full cache-miss rerun of LLM stages requires
the environment variables described in `llm/client.py`.

## Key Artefacts

- `experiments/ablation_results_sachs.json` — canonical Sachs ablation.
- `experiments/constraint_quality_sachs.json` — global and coverage-conditional
  constraint quality.
- `docs/step_06_5_stabilisation.md` — Sachs stabilisation verdict and paper
  framing.
- `data/dream4_psn/` and `experiments/*dream4_psn*` — Step 7 synthetic fallback
  after official DREAM4 PSN data required authenticated Synapse access.
- `paper/main.tex` — current paper draft scaffold.

## Reproduction Notes

All recurring commands are in `Taskfile.yml`. The most important targets are:

- `task ablation` — regenerate Sachs ablation, constraint quality, cost report,
  tables, and figures.
- `task lingam-sweep-sachs` — rerun the Step 6.5 sparse LiNGAM sweep.
- `task diagnose-pc-sachs` — rerun the PC invariance/AUPR diagnostic.
- `task dream4-psn` — regenerate the DREAM4 synthetic fallback dataset and
  results.
- `task test` — run the unittest suite.

## Citation

```bibtex
@misc{text2causal2026,
  title = {Auditable Reactome-Grounded Priors for Protein-Signalling Causal Discovery},
  author = {Jaroslawski, Mikolaj and Sepiolo, Dominik and Ligeza, Antoni},
  year = {2026},
  note = {Preprint in preparation}
}
```
