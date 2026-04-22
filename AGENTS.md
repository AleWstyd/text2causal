# AGENT.md

This file defines the default operating guide for coding agents working in this repository. Follow it for any implementation, debugging, refactor, or documentation task unless the user explicitly overrides part of it.

## Purpose

`text2causal` is a Python 3.12 Streamlit application that:

- loads the LUCAS dataset,
- extracts causal relations from free-form background text via OpenRouter Free Models Router (`openrouter/free`),
- converts those relations into prior-knowledge constraints,
- runs causal discovery algorithms (`PC`, `GES`, `LiNGAM`),
- compares unconstrained and constrained graph quality with evaluation metrics.

Agents should preserve that end-to-end flow unless the task explicitly asks for a behavioral change.

## Core Principles

- Keep changes small, local, and reversible.
- Prefer repo-established workflows over ad hoc commands.
- Preserve current architecture unless there is a clear maintenance or correctness reason to change it.
- Do not silently change scientific assumptions, thresholds, graph semantics, or model behavior.
- When changing behavior, update or add tests in the same task.
- Call out any limitation you could not verify, especially if it involves external APIs or Streamlit UI behavior.

## Source Layout

- `main.py`: Streamlit UI entrypoint and rendering layer.
- `app_pipeline.py`: orchestration layer for dataset loading, LLM extraction, algorithm dispatch, and metrics.
- `llm/`: prompt definitions and relation extraction via OpenRouter.
- `constraints/`: conversion and validation of extracted relations into prior knowledge.
- `causal_discovery/`: wrappers around the supported causal discovery algorithms.
- `evaluation/`: metric helpers used to compare graphs.
- `utils/`: dataset loading and graph utilities.
- `tests/`: unittest-based regression coverage.
- `data/lucas/`: bundled dataset inputs and background text.
- `Taskfile.yml`: canonical developer and agent workflow entrypoints.

## Standard Workflow

Prefer Task targets whenever they exist:

- `task install`: install and sync dependencies with `uv`.
- `task run`: launch the Streamlit app.
- `task test`: run the unittest suite.
- `task format`: format Python code with Ruff.
- `task lint`: run Ruff checks.

If a needed workflow is not available in `Taskfile.yml`, use the narrowest direct command possible and mention that no Task target existed.

## Environment

- Python is managed with `uv`.
- The Taskfile loads variables from `.env`.
- LLM extraction uses the OpenAI-compatible OpenRouter API (`openrouter/free`), so `OPEN_ROUTER_API_KEY` must be set for live LLM calls.
- Do not hardcode secrets, keys, filesystem-specific paths, or machine-specific assumptions.

## Editing Rules

- Follow the existing code style and current module boundaries.
- Prefer typed, explicit function signatures where the file already uses them.
- Keep UI logic in `main.py` and orchestration/business logic in `app_pipeline.py` or lower-level modules.
- Put constraint-building logic in `constraints/`, not in the UI layer.
- Put graph manipulation helpers in `utils/graph_utils.py` unless a new module is more coherent.
- Reuse existing helpers before adding new abstractions.
- Avoid speculative refactors that are not needed for the task.

## Testing Expectations

Run the following after code changes:

1. `task format`
2. `task lint`
3. `task test` when behavior, logic, data handling, or public functions changed

When UI-only text or documentation changes are made, `task test` is optional unless the touched code affects runtime behavior.

## Repo-Specific Guidance

- The default dataset is the bundled LUCAS dataset in `data/lucas/`.
- `extract_llm_constraints()` filters relations by confidence threshold and validates them against dataset variable names.
- `GES` constraints are currently applied post hoc as direct edge edits.
- `PC` and `LiNGAM` use native prior-knowledge pathways.
- Keep these algorithm-specific semantics intact unless the task explicitly asks to change them.
- If you change confidence handling, prior-knowledge validation, or graph metric computation, add tests for the edge cases introduced.

## External API and Model Changes

- The current LLM integration lives in `llm/extract_relations.py` (OpenRouter, model `openrouter/free`).
- Do not change the model name, prompt contract, response format, or JSON parsing behavior without a clear reason.
- If you do change any of those, update tests or add mocks around the new contract where feasible.
- Prefer deterministic, mockable seams for anything that would otherwise require live API calls in tests.

## Documentation Expectations

- Update this file when repo workflow or conventions materially change.
- Update `Taskfile.yml` when you introduce a recurring command that future agents should use.
- Keep README and agent docs aligned if setup or runtime expectations change.

## Final Handoff Checklist

Before closing a task, agents should:

- summarize the user-visible change,
- list verification actually run,
- mention any commands that could not be run,
- mention any external dependency or credential limitation that affected validation.
