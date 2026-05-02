# Step 4 — Causal Reasoning + OmniPath Floor

**Role.** Core. The central intellectual contribution.

**Goal.** For each pair of grounded variables, produce a directional causal claim with confidence — via LLM reasoning over Reactome reaction context (primary path) and via OmniPath direct-edge consensus (no-LLM floor for the C0.5 ablation). Also emit the LLM judgments directly as a predicted DAG (used for the `C-LLM-only` ablation; see Step 6).

**Depends on.** Step 2 (`ReactomeClient`), Step 3 (grounding).

**Effort.** 4–6 days. Most of the time is on prompt iteration, conflict aggregation, and the OmniPath integration — not the LLM call itself.

---

## Deliverables

| Artefact | Purpose |
|----------|---------|
| `reasoning/reason.py` | LLM-based causal direction inference |
| `reasoning/dag_from_priors.py` | Cycle-broken DAG emission from LLM claims (used by C-LLM-only) |
| `omnipath/floor.py` | No-LLM constraint set from OmniPath consensus |
| `experiments/causal_priors_sachs.json` | LLM-derived priors with full reasoning traces |
| `experiments/predicted_dag_llm_only_sachs.gml` | LLM-only predicted DAG for the C-LLM-only condition |
| `experiments/floor_priors_sachs.json` | OmniPath-derived no-LLM priors |
| Tests in `tests/test_reasoning.py`, `tests/test_floor.py`, `tests/test_dag_from_priors.py` | Aggregation, conflict resolution, cycle breaking |

---

## Output schemas

### LLM priors (`causal_priors_*.json`)

```json
{
  "pairs": [
    {
      "var_a": "praf",
      "var_b": "pmek",
      "cause": "praf",
      "effect": "pmek",
      "confidence": 0.94,
      "constraint_type": "hard_required",
      "supporting_reactions": ["R-HSA-111933", "R-HSA-5673001"],
      "contradicting_reactions": [],
      "reasoning": "RAF1 acts as catalyst in 2 phosphorylation reactions where MAP2K1 is output. No contradicting roles found.",
      "reactome_context_size": 4
    },
    {
      "var_a": "PIP2",
      "var_b": "plcg",
      "cause": "unknown",
      "effect": "unknown",
      "confidence": 0.45,
      "constraint_type": "unknown",
      "supporting_reactions": [],
      "contradicting_reactions": [],
      "reasoning": "Both appear as input to multiple reactions; direction not establishable from Reactome roles.",
      "reactome_context_size": 8
    }
  ],
  "no_context_pairs": [["P38", "PIP3"]],
  "model_id": "openrouter/free",
  "served_models": {"openai/gpt-oss-20b": 73, "qwen/qwen3-coder-14b-free": 37},
  "n_pairs_total": 110,
  "n_with_context": 87,
  "n_high_confidence": 23
}
```

### Floor priors (`floor_priors_*.json`)

```json
{
  "pairs": [
    {
      "var_a": "praf",
      "var_b": "pmek",
      "cause": "praf",
      "effect": "pmek",
      "confidence": 0.85,
      "constraint_type": "soft_prior",
      "n_sources": 4,
      "sources": ["Reactome", "SIGNOR", "KEGG", "PathwayCommons"],
      "is_directed": true,
      "consensus_sign": "activates"
    }
  ],
  "source_filter": "all" | "reactome_only"
}
```

---

## Implementation tasks

### LLM reasoning path

1. **Implement the prompt** from `docs/dev_plan_v2.md` §8.2. Lock the prompt before scaling — iterate on a handful of pairs first.

2. **Implement `reason_pair(...)`** in `reasoning/reason.py`:
   ```python
   def reason_pair(
       entity_a: EntityRef,
       entity_b: EntityRef,
       reactome_context: list[ReactionRecord],
       llm_client: LLMClient,
   ) -> CausalClaim: ...
   ```
   - If `reactome_context` is empty, return a `CausalClaim` with `constraint_type="no_context"`. Do not call the LLM. Save tokens.
   - Otherwise: format context via `format_context_for_llm`, build messages, call cached LLM client, parse JSON, validate against expected schema.

3. **Implement `reason_all_pairs(grounding, reactome_client, llm_client)`** that iterates ordered pairs:
   - For 11 Sachs variables: 11 × 10 = 110 ordered pairs (or 55 unordered if you prefer; ordered is recommended because it gives the LLM a chance to disagree with itself in opposite directions, which is informative).
   - Call `get_reaction_context` once per *unordered* pair, but call `reason_pair` twice (once per ordering) — the LLM may pick up direction differently when entity A vs B is highlighted.

4. **Aggregate per ordered pair.** When the LLM's two passes (A→B and B→A) disagree:
   - If both calls produce the same `cause`/`effect` and `constraint_type`, take the higher confidence.
   - If they conflict (different cause/effect with similar confidence), reduce final confidence to `min(c1, c2) * 0.7` — this is a soft penalty for instability.
   - Log all conflicts to `experiments/causal_priors_sachs.json` under `conflicts: [...]` for the paper.

5. **Apply the constraint-type decision rule:**
   - `confidence ≥ 0.9` → `hard_required` (or `hard_forbidden_reverse` if the LLM explicitly identified a forbidden direction).
   - `0.6 ≤ confidence < 0.9` → `soft_prior`.
   - `confidence < 0.6` → `unknown` (discarded for downstream use, but kept in the JSON for analysis).

6. **Persist `experiments/causal_priors_sachs.json`** with full reasoning traces. This file is committed and is auditable evidence.

### LLM-only DAG emission (for C-LLM-only condition)

7. **Implement `reasoning/dag_from_priors.py`.** Reads the just-persisted `causal_priors_sachs.json` and emits a single predicted DAG, no causal-discovery algorithm involved.
   ```python
   def predict_dag_from_priors(
       priors_path: Path,
       variable_names: list[str],
       confidence_threshold: float = 0.7,
   ) -> nx.DiGraph: ...
   ```
   Algorithm:
   - Filter priors to `confidence ≥ threshold` and `constraint_type ∈ {hard_required, soft_prior}`.
   - Sort the kept claims by confidence descending.
   - Initialise an empty `nx.DiGraph` with all `variable_names` as nodes.
   - For each claim in confidence-descending order:
     - Add the directed edge `cause → effect`.
     - If `nx.is_directed_acyclic_graph` returns False, remove the edge just added (it would have closed a cycle). Log the dropped edge to `dropped_due_to_cycle: [...]` for the paper.
   - Return the DAG.

8. **Persist `experiments/predicted_dag_llm_only_sachs.gml`** plus a sidecar JSON `experiments/predicted_dag_llm_only_sachs.meta.json` that records the threshold used, claims kept, claims dropped due to cycle, and the source priors hash.

### OmniPath floor path

9. **Install `pypath-omnipath`** (already added in Step 1).

10. **Implement `omnipath/floor.py`**:
   ```python
   def build_floor_priors(
       grounding: dict[str, GroundingRecord],
       source_filter: Literal["all", "reactome_only"] = "all",
   ) -> dict: ...
   ```
   - For each ordered pair of grounded variables, query OmniPath for direct A→B edges via `omnipath.interactions.AllInteractions.get(...)` (or equivalent — verify current API). Pass UniProt IDs.
   - Filter to `is_directed=True`.
   - When `source_filter == "reactome_only"`, restrict to interactions whose `sources` field contains `Reactome`.
   - Aggregate over the family / complex members: an edge counts if any member-to-member interaction exists.
   - Confidence = `min(1.0, n_sources / 3.0)`. Single-source = 0.33, two-source = 0.66, ≥3-source = 1.0.
   - Apply the same constraint-type decision rule as the LLM path.
   - Generate **two** floor files: `floor_priors_sachs_all.json` and `floor_priors_sachs_reactome_only.json`. The second is the apples-to-apples comparison with the LLM path; the first is the "any structured ontology" comparison.

### Smoke test before scaling

11. **Run on 5 known Sachs pairs first** (RAF→MEK, MEK→ERK, PKC→RAF, PI3K→Akt, PIP3→Akt). Inspect outputs by hand. If the LLM produces obviously wrong directions or low confidence on slam-dunk pairs, iterate the prompt before running the full sweep.

12. **Then run the full Sachs sweep.**

---

## Acceptance criteria

- [ ] Every Sachs ordered pair has a record in `causal_priors_sachs.json`, either a claim or a `no_context` flag.
- [ ] Reasoning traces include `supporting_reactions` and `contradicting_reactions` populated from real `R-HSA-*` IDs.
- [ ] Cache hit rate is 100% on second run.
- [ ] On the 5 smoke-test pairs, ≥ 4 have correct direction at confidence ≥ 0.7.
- [ ] `predicted_dag_llm_only_sachs.gml` exists, is acyclic, and contains all 11 Sachs variables as nodes (even if some are isolated).
- [ ] `predicted_dag_llm_only_sachs.meta.json` logs the count of claims dropped due to cycle (may be zero, may be small).
- [ ] OmniPath floor produces non-empty priors for both source filters; counts logged.
- [ ] `task lint` and `task test` pass.

---

## Pitfalls

- **The LLM may emit names outside the column-name vocabulary.** The prompt asks for `cause` and `effect` as `name_a` or `name_b`, but it might emit `RAF1` instead of `praf`. Validate strictly and reject (or remap) any output that doesn't match the input vocabulary.
- **Family members behave differently.** When the LLM sees PKC as a list, it may fixate on one isoform's role. The prompt's reasoning rules should explicitly mention "co-participation of any family member counts."
- **Same-input pairs are NOT direction-establishing.** Ensure the prompt rule about both-as-input is enforced. Periodically inspect pairs where the LLM violated it; that's a prompt bug.
- **OmniPath API drift.** `pypath-omnipath` evolves quickly; verify import paths and method signatures against the installed version. `omnipath` (the lighter REST client) and `pypath` are different packages with overlapping but distinct APIs.
- **OmniPath direction is incomplete.** Many sources don't carry direction. After filtering `is_directed=True`, a pair may have no edges even when one exists in the underlying database. This is fine — it's the floor's honest answer.
- **LLM cost on retry.** If you change the prompt mid-sweep, every cached call becomes invalid (different prompt hash). Iterate on smoke-test pairs first; do not mass-rerun.
- **Cycle-breaking ordering matters.** Adding edges in confidence-descending order is the standard greedy approach but will sometimes drop a high-confidence edge if it closes a cycle with two even-higher-confidence edges. Document any dropped edges; they're informative for the paper.

---

## Out of scope

- Constraint translation to `BackgroundKnowledge` / LiNGAM prior matrix (Step 5).
- Deterministic rule-based reasoner over Reactome subgraphs (parking lot; possible stretch goal in Step 6 if the OmniPath floor is too coarse).
- Discovery algorithm runs (Step 5).
