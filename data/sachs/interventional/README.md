# Sachs et al. (2005) interventional flow-cytometry panel

This directory contains the **nine experimental conditions** from Sachs *et al.*
(*Science* **308**, 523–529, 2005; DOI [10.1126/science.1105809](https://doi.org/10.1126/science.1105809)),
bundled as one CSV per condition plus a machine-readable `manifest.json`.

## Provenance

- **Primary publication:** K. Sachs, O. Perez, D. Pe’er, D. A. Lauffenburger, G. P.
  Nolan, *Causal protein-signaling networks derived from multiparameter single-cell
  data.* Science **308**, 523–529 (2005).
- **Files here** were taken from the Zenodo repackaging of the same measurements
  (Sachs flow-cytometry bundle; DOI [10.5281/zenodo.7681811](https://doi.org/10.5281/zenodo.7681811)),
  which consolidates the experimental CSVs under `Data Files/` in that archive.
  Column names follow the original export (`Raf`, `Mek`, `Erk`, …).
- The **R package bnlearn** also redistributes the Sachs example graph and related
  tooling under the GPL-3 licence; we cite it as an additional community mirror,
  even though these bytes were copied from the Zenodo bundle rather than from R’s
  binary `.rda` objects.

## Layout

- `raw/*.csv` — one file per perturbation context; rows are single cells, columns
  are the 11 phospho-readouts used throughout this repository (after renaming to
  match `load_sachs_dataset()`; see `utils/load_data.py`).
- `manifest.json` — per-condition metadata: sample counts, nominal perturbation
  agent, and `gies_target_indices` (0-based column indices in the canonical
  Sachs column order used by this repo) passed to GIES as interventional targets.

## Licence / redistribution

Science supplementary materials and downstream redistributions (Zenodo, toolkits)
have supported open redistribution for benchmarking. Retain both the Science
citation and the Zenodo DOI when re-publishing derivatives.
