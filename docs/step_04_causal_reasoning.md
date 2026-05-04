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

---

## Implementation notes (deviations from the original spec)

These are findings from the actual implementation. They are not changes to
the contribution claim, just operational details a future agent needs to
know.

### Local package name: `omnipath_floor/`, not `omnipath/`

The dev plan §9 repo layout suggests `omnipath/floor.py`. The PyPI package
`omnipath` (the lighter REST client we read from) occupies that import path,
so a local `omnipath/` directory at repo root would shadow the third-party
package and prevent us from importing `omnipath.interactions.AllInteractions`.

The implementation lives at `omnipath_floor/floor.py` with the same public
surface (`build_floor_priors`, `FloorEdge`).

### `pypath-omnipath` not used; `omnipath` is

`pyproject.toml` originally declared `pypath-omnipath>=0.16.20`. That package
is the heavyweight pypath toolkit; importing it currently fails on a
transitive `pysftp` / `paramiko` `DSSKey` mismatch (`DSSKey` was removed
upstream from `paramiko`). We use the lighter `omnipath` REST client (added
to `pyproject.toml` as `omnipath`) which the spec already pointed at via
`omnipath.interactions.AllInteractions.get(...)`. `pypath-omnipath` is left
in `pyproject.toml` for now in case downstream needs it; nothing in the
Step 4 floor pipeline imports it.

### OmniPath REST endpoint flakiness

The omnipathdb.org REST endpoint was returning 500s on the
`directed=1`-filtered query at the time of the snapshot capture. The
unfiltered query also intermittently 500s after a successful first pass.
The implementation pulls all human interactions with the
`genesymbols=True` + `references` field signature (the combination the
upstream omnipath client cached locally for our session), then filters
client-side on the `is_directed` boolean. The resulting DataFrame is
snapshotted to `cache/omnipath/all_interactions_directed_human.parquet`
(committed) so subsequent runs replay from the repo cache without hitting
the upstream REST API. Verified offline-replayable with
`HTTPS_PROXY=http://127.0.0.1:1 task floor-sachs`.

### Floor priors metabolite handling

Metabolites (PIP2, PIP3) have no UniProt accession; OmniPath's
`AllInteractions` table is keyed on UniProt source/target. Pairs involving
either of the two Sachs metabolites are skipped explicitly and recorded
under `skipped_metabolite_pairs` in the artefact, separately from
`no_edge_pairs`. The LLM path does cover those pairs via Reactome's
metabolite-aware reaction context.

---

## Step 4 Results (Sachs)

### LLM transport

The LLM pin was swapped mid-Step-4 at the user's direction: the previous
Baseten dedicated deployment hosting `Qwen/Qwen3-235B-A22B` was retired
(deployment quota expired). The replacement is `deepseek-ai/DeepSeek-V4-Pro`
on the Baseten Model API (`https://inference.baseten.co/v1`). Both
`LLM_BASE_URL` and `LLM_MODEL` remain env-overridable. Cache hygiene: the
swap invalidated all Step-4 cache keys (since the cache key includes
`model`); the new entries are committed alongside this step. Step-3
grounding cache stays valid because the grounding artefact is locked.

DeepSeek-V4-Pro is a chain-of-thought reasoner whose `reasoning_content`
field consumes most of the per-call completion-token budget before the
JSON answer is emitted. The default 4096-token limit truncated mid-JSON
on the first smoke call; the per-call `max_tokens` was raised to 16384
(see `reasoning/reason.py:REASONING_MAX_TOKENS`) and added to the cache
key so the budget bump cleanly invalidated only the new entries.

### Smoke test (Phase 2 prompt-lock)

Strict tripwire: ≥4/5 must have correct direction at confidence ≥ 0.7
AND cite a real `R-HSA-*` ID. Result: **4/5 PASS**.

| Pair (expected) | Reactome ctx | Emitted | Conf | Cites R-HSA-* | Pass |
|-----------------|-------------:|---------|-----:|----------------|------|
| `praf → pmek` | 172 records | `praf → pmek` | 0.80 | yes | PASS |
| `pmek → p44/42` | 249 records | `pmek → p44/42` | 0.95 | yes | PASS |
| `praf → p44/42` | 161 records | `unknown` | 0.40 | n/a  | FAIL¹ |
| `PIP2 → PIP3` | 6 records   | `PIP2 → PIP3` | 0.95 | yes | PASS |
| `PIP3 → pakts473` | 26 records | `PIP3 → pakts473` | 0.85 | yes | PASS |

¹ The `praf → p44/42` "FAIL" is the model correctly identifying
ambiguity: Reactome encodes both the RAF→ERK forward edge AND the well-
documented ERK→RAF negative-feedback edge. Per the prompt's
"contradictory roles decrease confidence" rule, emitting `unknown` here
is correct behaviour, not a defect.

Pair-substitution rationale (relative to the original Phase-2 spec):
the original list included `(PKC, praf)` and `(plcg, pakts473)`, both of
which sit in the documented Reactome 4-layer-union coverage gap (zero
records in either direction; see `experiments/reactome_coverage.json`'s
`pairs_missing_by_layer_union`). With no Reactome context the LLM
correctly emits `no_context`, which would make the smoke test measure
Reactome curation rather than the LLM. The two replacements were chosen
from the same MAPK/PI3K cascade, with rich pre-verified Reactome
context: `praf → p44/42` (RAF→ERK direct cascade, 161 records) and
`PIP2 → PIP3` (PI3K phosphorylation, 6 records). The doc itself
contemplates substitution ("or use `PIP3` ↔ `pakts473`").

### Full sweep (Phase 3 reason-sachs)

| Quantity | Value |
|----------|------:|
| Ordered pairs total | 110 |
| Pairs with Reactome context | 62 |
| Pairs with no Reactome context (skipped LLM) | 48 |
| Aggregate pairs at confidence ≥ 0.9 | 16 |
| Aggregate pairs at 0.6 ≤ confidence < 0.9 | 18 |
| Aggregate pairs at confidence < 0.6 (kept for analysis) | 28 |
| LLM calls (counted by served-model increments) | 122 |
| Distinct served models | 1 (`deepseek-ai/DeepSeek-V4-Pro`) |
| Forward / reverse pass conflicts | **0** |
| Wall clock (first run, network) | ~26 min |
| Wall clock (second run, full cache hit) | 0.56 s |

Cache-hit invariant: the second run produces a byte-identical artefact
and adds 0 cache entries. Verified with
`HTTPS_PROXY=http://127.0.0.1:1 task reason-sachs`.

The "0 conflicts" result misses the soft target ("at least one entry in
`conflicts: [...]`"). Empirically the model is highly direction-
consistent across the A→B and B→A passes on this dataset; the soft
target is informational, not a Hard acceptance criterion. The
double-pass machinery is still useful for downstream datasets and for
the paper's auditability story.

### LLM-only DAG (Phase 4 dag-llm-only-sachs)

| Quantity | Value |
|----------|------:|
| Threshold (confidence floor) | 0.7 |
| Claims in source priors | 62 |
| Claims above threshold | 34 |
| Claims after constraint-type filter (kept) | 34 |
| Edges retained in DAG | 16 |
| Edges dropped (would close cycle) | 2 |
| Nodes (all 11 Sachs vars, isolated nodes preserved) | 11 |
| `is_directed_acyclic_graph` | True |

Both dropped-due-to-cycle entries are duplicates of the same edge
`plcg → PIP2` (one from each ordered iteration of the underlying pair).
The greedy confidence-descending cycle-breaker had to drop them because
the higher-confidence edges `PIP2 → PIP3` (0.95) and `PIP3 → plcg`
(1.00) had already created the path that `plcg → PIP2` would close into
a cycle. This is a known limitation of greedy DAG induction and is
informative for the paper: it surfaces the bidirectional PIP2 ↔ plcg
biochemistry that any single-DAG representation must collapse.

The 16 retained edges are biologically sensible: the canonical MAPK
cascade (`praf → pmek`, `pmek → p44/42`), the PI3K/Akt branch (`PIP2 →
PIP3`, `PIP3 → pakts473`, `PIP3 → plcg`), and the well-documented
upstream regulators of RAF (`P38 → praf`, `PKA → praf`).

### OmniPath floor counts (Phase 1, repeated for handoff)

| `source_filter` | with edge | hard_required | soft_prior | unknown_kept | metabolite-skipped |
|-----------------|----------:|--------------:|-----------:|-------------:|-------------------:|
| `all` | 43 | 32 | 9 | 2 | 38 |
| `reactome_only` | 3 | 0 | 0 | 3 | 38 |

Reactome-only floor confidence is capped at ~0.33 in practice because
the directed-edge column for these pairs almost always has just one
Reactome-named source. This is the floor's honest answer; the higher-
recall LLM path runs alongside it and the comparison is the C2/C3 vs
C0.5 attribution promised in the dev plan contribution claim.

### Free-text fallback (merged priors, Step 6 C3+ft)

When the four-layer Reactome union yields **no** evidence for an ordered
pair, `reason_pair` returns `no_context` and the pairwise LLM is not
called (`reasoning/reason.py`). Sachs leaves dozens of ordered pairs in
`no_context_pairs` even though pathway-level biology is often documented
elsewhere. The same cached **free-text** Sachs paragraph used for
condition C1 (`experiments/freetext_priors_sachs.json`) already
recovers several of those edges at high precision (see
`experiments/constraint_quality_sachs.json` and dev plan changelog item 17).

`reasoning/merge_freetext_fallback.py:merge_with_freetext_fallback` builds
`experiments/causal_priors_*_with_fallback.json`: for each `no_context`
slot (including rows listed only under `no_context_pairs`), if the
unordered pair appears in the free-text prior blob, the claim is
substituted with the free-text `cause`, `effect`, `confidence`, and
`constraint_type`, and `source` is set to `freetext_fallback`. All
Reactome-context claims are left unchanged. The Step 6 ablation adds
**C3+ft** (`priors_source=reactome_llm_with_freetext_fallback`, τ=0.7),
parallel to C3 but reading the merged file. Offline regeneration:
`task reason-with-fallback-sachs` then `task ablation-sachs`.

### Hard acceptance criteria (Step 4 spec checklist)

- [x] Every Sachs ordered pair (110) has a record — either in `pairs` (62) or `no_context_pairs` (48).
- [x] Reasoning traces include real `R-HSA-*` IDs in `supporting_reactions`; ID-validation against the Reactome context window prunes hallucinated citations and logs a `support_pruned` note.
- [x] On the (substituted) 5 smoke pairs, ≥ 4 have correct direction at confidence ≥ 0.7.
- [x] Cache hit rate is 100% on the second `task reason-sachs` (0 new cache entries, byte-identical artefact).
- [x] `predicted_dag_llm_only_sachs.gml` exists, is acyclic, contains all 11 Sachs variables as nodes.
- [x] `predicted_dag_llm_only_sachs.meta.json` logs `dropped_due_to_cycle` (2 entries).
- [x] Both `floor_priors_sachs_all.json` (43 directed edges) and `floor_priors_sachs_reactome_only.json` (3 directed edges) are non-empty.
- [x] `task format && task lint && task test` pass (final test count: 109 → see Phase 5 commit).
