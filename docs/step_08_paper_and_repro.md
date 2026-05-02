# Step 8 — Paper + Reproducibility Wrap

**Role.** Evaluation. Final.

**Goal.** Paper draft (Introduction → Discussion) and a fully-reproducible public repository where a clean clone can regenerate every reported number with one command.

**Depends on.** Steps 6 and 7 — all results in hand, no further pipeline changes.

**Effort.** 2–3 weeks. The single most underestimated step in v1's plan.

---

## Deliverables

| Artefact | Purpose |
|----------|---------|
| `paper/main.tex` | Compilable LaTeX paper |
| `paper/refs.bib` | BibTeX bibliography |
| `paper/figures/` | Symlinked or copied from `figures/` |
| `paper/tables/` | Symlinked or copied from `tables/` |
| `paper/appendix.tex` | Reproducibility appendix (caches, served-model breakdown, prompts, REST endpoints used) |
| `README.md` | Project overview, install instructions, reproduction steps |
| `LICENSE` | License file (recommended: MIT or Apache 2.0) |
| Updated `Taskfile.yml` | `task install`, `task ablation`, `task paper` |
| Public GitHub repo | Final commit, tagged for paper version |
| arXiv preprint | Submitted before venue submission |

---

## Implementation tasks

### Week 1 — Code freeze + reproducibility

1. **Freeze the pipeline.** No further behavioral changes after this point. Bug fixes only, and only if they affect paper numbers.

2. **Pin every dependency.** Update `pyproject.toml` to exact pinned versions (`==`, not `>=`). Run `task install` from a clean clone to verify.

3. **Document the Reactome API contract.** Record the exact REST endpoints used, with example URLs, in the paper appendix. The committed `cache/reactome/` provides bytes-for-bytes replay regardless of any future API change.

4. **Document the LLM model setup.** OpenRouter `openrouter/free` is a routing alias — the paper appendix reports the per-request `served_model` distribution (e.g. "37% openai/gpt-oss-20b, 63% qwen/qwen3-coder-14b-free over the date range X–Y") aggregated from the cache. Also report total tokens and total cost (likely $0 for free-tier).

5. **Commit the LLM cache.** All `cache/llm/**/*.json` files. This is the single most important reproducibility artefact — it lets reviewers replay the pipeline without API access.
   - Sanity check: in a clean clone with `OPEN_ROUTER_API_KEY` unset, `task ablation` should still complete from cache.

6. **Commit all priors files** (`experiments/grounding_*.json`, `experiments/causal_priors_*.json`, `experiments/floor_priors_*.json`, `experiments/oracle_priors_*.json`, `experiments/freetext_priors_*.json`).

7. **Write `README.md`.** Sections:
   - One-paragraph summary.
   - Quickstart: `task install && task ablation` (no Docker; the committed cache replays the pipeline offline).
   - Reproduction guarantee statement (clean-clone, no API key, all results regenerable from cache).
   - Project layout.
   - Citation block.

8. **Add `task paper`** that compiles `paper/main.tex` from cached results — no pipeline run required.

### Week 1–2 — Paper draft

9. **Introduction (1–1.5 pages).**
   - Motivation: implicit LLM knowledge is unverifiable.
   - Three claims (auditable, generalises, measurable).
   - Contributions list.

10. **Related Work (0.5–1 page).** Use the table from `dev_plan_v2.md` §1.2. Cite CausalCopilot, ALCM, CATE-B explicitly.

11. **Method (2–3 pages).**
    - Architecture overview (4 stages).
    - Reactome schema (the corrected one — input/output direct, catalystActivity 2-hop, regulation 2-hop).
    - Variable grounding (UniProt + ChEBI + family lists).
    - Causal reasoning prompt and decision rules.
    - Constraint integration per algorithm (PC native, LiNGAM matrix, GES post-hoc with caveat).
    - OmniPath floor.
    - Reference appendix for full prompts.

12. **Experiments (1–1.5 pages).**
    - Datasets: Sachs primary, DREAM4 PSN secondary (or synthetic-from-Reactome if Step 7 tripwire fired).
    - Ablation conditions table (C0, C0.5, C1, C2, C3, C4, C5).
    - Metrics (SHD, AUPR, P, R, F1).
    - Constraint quality metrics (precision, recall, hallucination, coverage).
    - Reproducibility statement (cache, pinned everything).

13. **Results (1.5–2 pages).**
    - Baseline performance (C0).
    - Improvement per condition (C0.5, C1, C2, C3, C4) per algorithm, both datasets.
    - **Headline number: % of C0→C5 oracle gap closed by best Reactome+LLM condition, per algorithm, per dataset.**
    - Constraint quality table.
    - Threshold sensitivity figure.
    - Cost report (tokens, $, wall-clock).
    - Note any surprising findings.

14. **Discussion (1–1.5 pages).**
    - Why did certain algorithms benefit more (LiNGAM's directional priors vs PC's CPDAG vs GES's post-hoc edits)?
    - Hallucination rate in context: is the system safe to use?
    - C2/C3 vs C0.5: does the LLM actually contribute over OmniPath consensus? Quantified answer.
    - C2/C3 vs C1: does Reactome grounding add value over plain free-text-LLM? Quantified answer.
    - Threshold sensitivity: is τ = 0.7 robust?
    - Position vs CausalCopilot, ALCM, CATE-B.
    - Limitations: two datasets in one biology family, English-only ontology, single LLM family, GES post-hoc caveat.
    - Future work: cross-domain (clinical, epidemiology); deterministic rule-based reasoner over Reactome; cross-ontology comparison (KEGG, SIGNOR alone).

15. **Reproducibility appendix.**
    - Pinned versions table.
    - LLM setup: `openrouter/free` routing alias; `served_model` distribution aggregated from cache; total tokens; total cost; date range.
    - Reactome access: REST API base URL, endpoints used, cache layout.
    - Full grounding and causal-reasoning prompts.
    - Cache layout description.
    - One-command reproduction instructions.

### Week 2–3 — Polish + submission

16. **Internal review.** Get one collaborator (Sepioło or Ligęza) to read the draft. Iterate.

17. **arXiv preprint.** Submit before venue submission so it has a stable DOI and can be cited.

18. **Public GitHub repo.** Push everything; tag the commit (e.g. `paper-v1`).

19. **Submit to chosen venue.** Per `dev_plan_v2.md` §5: CLeaR 2027 primary, ECAI 2027 fallback, RAI 2026 if Steps 1–6 finish in time.

20. **Update `dev_plan_v2.md` changelog** with the actual venue + submission date.

---

## Acceptance criteria

- [ ] `paper/main.tex` compiles cleanly with no LaTeX warnings.
- [ ] Every numerical claim in the paper is traceable to a JSON artefact (`ablation_results_*.json`, `constraint_quality_*.json`, `cost_report.json`).
- [ ] On a clean clone with `OPEN_ROUTER_API_KEY` unset, `task install && task ablation` completes successfully from cache and reproduces all reported numbers within rounding.
- [ ] `README.md` quickstart works on a fresh machine (verified by a colleague, ideally).
- [ ] Public GitHub repo is up; arXiv preprint is up.
- [ ] Venue submission is confirmed.

---

## Pitfalls

- **Don't change pipeline behavior during writing.** Bug fixes that affect numbers force a full rerun; bug fixes that only affect reporting are fine. Be disciplined about which is which.
- **Numerical inconsistency.** The single most common paper bug: numbers in prose disagree with numbers in tables because one of them was hand-edited. Source every number from the JSON artefacts via a script, not by hand-typing.
- **Cache size.** `cache/llm/` may be tens of MB. Commit it anyway. Use Git LFS if it crosses ~100 MB.
- **Reproducibility statement scope.** Be honest: "all reported numbers reproducible from cache; full end-to-end re-runs require an API key and produce slightly different LLM outputs." Don't overclaim determinism you don't have.
- **Bibliography drift.** Use BibTeX entries from official sources (DBLP, journal pages), not the author's hand-typed approximations.
- **Anonymisation.** If submitting to a double-blind venue, anonymise the GitHub repo and remove author names from the arXiv preprint until acceptance.

---

## Out of scope

- Camera-ready post-acceptance edits.
- Journal extension.
- Code changes that would invalidate paper numbers.
