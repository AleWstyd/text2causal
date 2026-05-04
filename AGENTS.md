# AGENTS.md

This file is the default operating guide for coding agents working in this
repository. Follow it for any implementation, debugging, refactor, or
documentation task unless the user explicitly overrides part of it.

## Purpose

`text2causal` is a Python 3.12 research pipeline that derives causal-discovery
priors for protein-signalling datasets by combining a structured ontology
(Reactome) with LLM reasoning, and then runs constrained causal discovery
(`PC`, `GES`, `LiNGAM`) and reports paper-grade evaluation metrics.

The work is organised as the multi-step plan in `docs/dev_plan_v2.md`. Every
step has a per-step spec under `docs/step_NN_*.md` (the authoritative scope
for that step). When a task names "Step N", read `docs/step_0N_*.md` first
and `docs/dev_plan_v2.md` second; treat the per-step doc as ground truth on
scope, deliverables, and acceptance criteria.

A legacy Streamlit / LUCAS skeleton (`main.py`, `app_pipeline.py`, the
`llm/extract_relations.py` free-text extractor, the LUCAS dataset under
`data/lucas/`) is preserved deliberately as the **C1 free-text-LLM baseline**
(see `docs/dev_plan_v2.md` §3 Step 6 ablations and §7 parking lot). Do not
delete it. Do not extend it for new pipeline work either — new functionality
goes into the per-step modules below.

## Core Principles

- Keep changes small, local, and reversible.
- Prefer repo-established workflows (`Taskfile.yml` targets) over ad hoc commands.
- Preserve current architecture and module boundaries unless the task explicitly says otherwise.
- Do not silently change scientific assumptions, thresholds, graph semantics, prompt contracts, or model behavior.
- When changing behavior, update or add tests in the same task.
- The pipeline is **offline-replayable by design**. Anything that calls an external service must go through `llm/cache.py:cached_call` or its Reactome wrapper, and the resulting JSON cache files must be committed in the same task.
- Call out any limitation you could not verify, especially anything depending on a live external API.

## Source Layout

Active research pipeline (this is where new work goes):

- `reactome/` — Reactome Content Service REST client (`client.py`) and endpoint reference (`endpoints.md`). Built in Step 2 (extended in Step 2.5 with co-pathway / regulator-chain / co-complex layers).
- `llm/cache.py` — single-JSON-per-key on-disk cache used by both the LLM client and the Reactome client. **All external calls funnel through this.**
- `llm/client.py` — OpenRouter chat-completion wrapper that records the per-request `served_model`. The model identifier is pinned in `llm/client.py:MODEL`; do not hardcode it in other modules.
- `llm/prompts.py` — shared prompt templates used by Step 3+ (research pipeline).
- `grounding/` — Step 3 variable grounding (LLM-driven dataset-column → UniProt / ChEBI / family mapping with Reactome validation). `ground.py` is the entry point; `evaluate.py` is the gold-standard scoring helper.
- `evaluation/` — `harness.py` evaluates a predicted graph against ground truth; `metrics.py` carries the primitives.
- `constraints/constraint_builder.py` — translates priors into PC `BackgroundKnowledge`, LiNGAM prior matrices, and post-hoc GES edits. Will grow in Step 5.
- `causal_discovery/` — algorithm wrappers (`run_pc.py`, `run_ges.py`, `run_lingam.py`).
- `experiments/` — runnable scripts and their JSON outputs (`baseline_sachs.py`/`.json`, `reactome_coverage.py`/`.json`, `run_grounding_sachs.py`, `grounding_sachs_gold.json`, `grounding_dream4.json`, `run_discovery_sachs.py`/`.json` Step 5 Sachs sweep, `runner.py` Step 6 ablation sweep → `ablation_results_sachs.json`, `constraint_quality.py` Step 6 Phase 3 → `constraint_quality_sachs.json`, `report.py` Step 6 Phase 4 → LaTeX under `tables/` and figures under `figures/`, `cost_report.py` Step 6 Phase 5 → `cost_report.json`). Most scripts write one committed JSON artefact; the Step 5 sweep persists incrementally to `experiments/discovery_results_sachs.json`; Step 6 persists incrementally to `experiments/ablation_results_sachs.json`.
- `tables/` — Step 6 Phase 4 paper `tabular` snippets (`ablation_table.tex`, `constraint_quality.tex`).
- `figures/` — Step 6 Phase 4 matplotlib outputs (each plot as `.pdf` and `.png`).
- `utils/load_data.py` — dataset loaders (`load_sachs_dataset`, `load_lucas_dataset`).
- `utils/graph_utils.py` — graph-manipulation helpers.
- `tests/` — unittest-based regression coverage. Fixtures live in `tests/fixtures/`.
- `cache/` — committed response caches (`cache/llm/`, `cache/reactome/`). The pipeline replays end-to-end from these without API access.

Legacy / C1 baseline (do not extend, do not delete):

- `main.py` — Streamlit UI for the LUCAS skeleton.
- `app_pipeline.py` — orchestration for the LUCAS skeleton.
- `llm/extract_relations.py` — free-text relation extractor used by the C1 ablation only.
- `data/lucas/` — bundled LUCAS dataset.

Project documentation:

- `docs/dev_plan_v2.md` — full multi-step plan; **read this for cross-step context.**
- `docs/step_0N_*.md` — per-step specs; **read the relevant one first when given a Step-N task.**

## Standard Workflow

Prefer `Taskfile.yml` targets when they exist:

- `task install` — install / sync dependencies via `uv`.
- `task test` — run the unittest suite (offline; should pass with `HTTPS_PROXY=http://127.0.0.1:1`).
- `task format` — Ruff format.
- `task lint` — Ruff check.
- `task baseline-sachs` — Sachs unconstrained PC/GES/LiNGAM baseline → `experiments/baseline_sachs.json`.
- `task discover-sachs` — Step 5 Sachs constrained-discovery sweep → `experiments/discovery_results_sachs.json`.
- `task ablation-sachs` — Step 6 canonical ablation sweep on Sachs → `experiments/ablation_results_sachs.json`.
- `task constraint-quality-sachs` — Step 6 constraint quality (priors vs ground truth) → `experiments/constraint_quality_sachs.json`.
- `task report-sachs` — Step 6 Phase 4 reporting → `tables/*.tex`, `figures/*.{pdf,png}` from committed JSON artefacts.
- `task cost-report` — Step 6 Phase 5 LLM cache token/cost summary → `experiments/cost_report.json`.
- `task ablation` — Step 6 runner + constraint quality + cost report + report (full ablation pipeline).
- `task reactome-coverage` — Sachs (+ DREAM4 smoke) Reactome coverage probe → `experiments/reactome_coverage.json`.
- `task ground-sachs` — batched LLM grounding for Sachs columns → `experiments/grounding_sachs.json`.
- `task oracle-priors-sachs` — Step 6 Phase 1 oracle ground-truth priors for Sachs → `experiments/oracle_priors_sachs.json`.
- `task freetext-sachs` — Step 6 Phase 1 C1 free-text LLM priors for Sachs → `experiments/freetext_priors_sachs.json` (uses committed `cache/llm/` on replay).
- `task run` — launches the **legacy Streamlit LUCAS skeleton** (kept for the C1 ablation only; not the research pipeline entrypoint).

If a needed workflow is not in `Taskfile.yml`, use the narrowest direct command possible and mention that no Task target existed. If the workflow is recurring and a future agent will run it, add a Task target in the same PR.

## Environment

- Python 3.12; dependencies managed by `uv` (`pyproject.toml` + `uv.lock`).
- `Taskfile.yml` loads variables from `.env`.
- LLM calls go through `llm/client.py` against an OpenAI-compatible API. The current pin is `Qwen/Qwen3-235B-A22B` served from a Baseten dedicated deployment; `BASETEN_API_KEY` (and optionally `LLM_BASE_URL` / `LLM_MODEL` for overrides) must be set in `.env` for a *first-time* (cache-miss) run; cache-hit runs are offline. `served_model` is recorded per cache entry for the paper appendix. The earlier `openrouter/free` alias and the Nemotron pin are obsolete (see dev plan changelog item 5 / §8 for the trail).
- Reactome calls use the public Content Service REST API via `reactome/client.py`. No credentials required. First run populates `cache/reactome/`; subsequent runs are offline.
- Do not hardcode secrets, API keys, filesystem-specific paths, or machine-specific assumptions.

## Editing Rules

- Follow the existing code style and current module boundaries.
- Prefer typed, explicit function signatures; the new modules are fully type-annotated.
- For new research-pipeline work:
  - Reactome / structured-knowledge access → `reactome/`.
  - LLM I/O and caching → `llm/`.
  - Variable grounding → `grounding/`.
  - Causal-prior translation → `constraints/`.
  - Experiment runners → `experiments/` (one script per artefact).
  - Tests → `tests/` with cached fixtures under `tests/fixtures/<module>/`.
- Do **not** add new pipeline logic to `main.py` or `app_pipeline.py` — those belong to the C1 legacy path.
- Reuse existing helpers (`cached_call`, `_resolve_regulation`, `_matches_for_pe`, `_pick_pathway`, evaluation harness, etc.) before adding new abstractions.
- Avoid speculative refactors that are not needed for the task.
- Every external service call goes through a cache wrapper. Cache files are committed as part of the task that creates them; they are not gitignored.

## Testing Expectations

After code changes, run:

1. `task format`
2. `task lint`
3. `task test` when behavior, logic, data handling, or public functions changed.
4. For tasks touching `reactome/`, `llm/`, `grounding/`, or any cache-using script, additionally verify offline replay: `HTTPS_PROXY=http://127.0.0.1:1 HTTP_PROXY=http://127.0.0.1:1 task <relevant-target>` must succeed.

When changes are documentation-only or touch only the legacy Streamlit path without modifying shared modules, `task test` is optional.

Test discipline for new code:

- Pre-seed a `TemporaryDirectory` cache via the existing `_CacheSeeder` pattern in `tests/test_reactome_client.py`.
- Patch `httpx.get` (or the OpenAI client) and assert it is **not** called when the test is supposed to run from cache.
- New fixtures under `tests/fixtures/` should each be < 2 KB; they are tests, not data dumps.

## Repo-Specific Guidance

- The primary research dataset is **Sachs** (`utils.load_data.load_sachs_dataset`); the secondary is **DREAM4 PSN** (Step 7). LUCAS is legacy.
- `experiments/reactome_coverage.json` records the live coverage figures and the explicit `escalation_decision` (whether to fall back to OmniPath as the primary structured source). Risk #2 in the dev plan is keyed to **node** coverage, not pair coverage.
- `extract_llm_constraints()` (legacy LUCAS path) filters relations by confidence threshold and validates them against dataset variable names. Do not change its semantics; the Step-3+ research pipeline does not use it.
- `PC` and `LiNGAM` use native prior-knowledge pathways in `causal-learn`. `GES` constraints are applied **post hoc** as direct edge edits — this is a known methodological caveat to be reported, not silently fixed (see dev plan Risk #6).
- LiNGAM prior-matrix convention: `1` required, `0` forbidden, `-1` unknown (default fill). Do not change this.
- If you change confidence handling, prior-knowledge validation, graph metric computation, or any cached prompt / response contract, add tests for the edge cases introduced.

## External API and Model Changes

- The research-pipeline LLM integration lives in `llm/client.py` + `llm/cache.py`. The legacy free-text extractor `llm/extract_relations.py` is kept only for the C1 baseline.
- Do not change the LLM model identifier, prompt templates, response format, or JSON parsing behavior without a clear reason. If you do, update the corresponding tests, regenerate the affected cache entries (and commit them), and note the change in the PR description.
- Prefer deterministic, mockable seams for anything that would otherwise require live API calls in tests. Live-network tests are not acceptable.
- The Reactome client must keep its existing public surface (`get_reaction_context`, `EntityRef`, `ReactionRecord`, `format_context_for_llm`, retry-with-backoff). Step 2.5 added per-pair evidence-layer methods alongside it; new layers (if any) should follow the same additive pattern.

## Documentation Expectations

- Update this file when repo workflow, module boundaries, or conventions materially change.
- Update `Taskfile.yml` when you introduce a recurring command that future agents should use, in the same PR.
- Keep `docs/dev_plan_v2.md` and the per-step `docs/step_NN_*.md` aligned with what shipped. If a step's implementation reveals a finding the spec missed, edit the spec in the same PR (or a small follow-up doc PR if the user has flagged docs as out of scope for the implementation task).
- README.md is currently empty. If you build something a new contributor needs to bootstrap, start it; otherwise leave it.

## Final Handoff Checklist

Before closing a task, agents should:

- summarise the user-visible change;
- list verification actually run (which `task` targets, which offline-replay checks);
- mention any commands that could not be run, and why;
- mention any external dependency or credential limitation that affected validation;
- confirm `git status` is clean and any new cache files are committed alongside the code that produced them.
