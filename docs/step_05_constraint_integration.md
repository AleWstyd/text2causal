# Step 5 — Constraint Integration and Constrained Discovery

**Role.** Core.

**Goal.** Translate causal priors (LLM and OmniPath floor) into the constraint formats expected by `causal-learn`'s PC, GES, and LiNGAM, and run constrained discovery for every `(condition × algorithm × seed)` combination on Sachs.

**Depends on.** Step 1 (baseline harness), Step 3 (grounding), Step 4 (priors).

**Effort.** 2 days.

---

## Deliverables

| Artefact | Purpose |
|----------|---------|
| `constraints/builder.py` | Extends current `constraint_builder.py` with threshold sweep + GES post-hoc explicit method |
| `experiments/run_condition.py` | Runs one `(dataset, priors_source, algorithm, threshold, seed)` cell |
| `experiments/discovery_results_sachs.json` | Per-cell results across the full sweep |
| Tests in `tests/test_builder.py` | Threshold sweep, conflict detection, all three algorithms |

---

## Implementation tasks

1. **Extend `constraints/builder.py` (rename file from `constraint_builder.py` if desired, keep backward-compatible imports).**
   ```python
   class ConstraintBuilder:
       def __init__(
           self,
           priors: list[dict],
           variable_names: list[str],
           confidence_threshold: float,
       ) -> None: ...

       def filtered_priors(self) -> tuple[list[Edge], list[Edge]]:
           """Return (required, forbidden) edges at the threshold."""

       def build_pc_background_knowledge(self) -> BackgroundKnowledge: ...
       def build_lingam_prior_matrix(self) -> np.ndarray: ...
       def build_ges_post_hoc_edits(self) -> tuple[list[Edge], list[Edge]]:
           """Return (edges_to_add, edges_to_remove). Naming advertises the caveat."""

       def summary(self) -> dict:
           """Counts of hard/soft/forbidden/discarded at the current threshold."""
   ```

2. **Reuse the existing PC and LiNGAM logic** from `constraints/constraint_builder.py`. The current implementation is correct for the v1 simple case; we just add the threshold sweep and the GES method.

3. **GES post-hoc method.** The current code applies constraints via `utils/graph_utils.apply_constraints` after GES finishes. Wrap that in `build_ges_post_hoc_edits` and have `experiments/run_condition.py` call it explicitly when `algorithm == "GES"`. This makes the caveat structurally visible in the code.

4. **Implement `experiments/run_condition.py`.**
   ```python
   def run_condition(
       *,
       dataset_name: str,
       data: np.ndarray,
       variable_names: list[str],
       true_graph: nx.DiGraph,
       priors: list[ClaimRecord] | None,
       priors_source: str,  # "none" | "reactome_llm" | "omnipath_all" | "omnipath_reactome_only" | "oracle"
       algorithm: str,  # "PC", "GES", "LiNGAM"
       threshold: float | None,  # None for C0 and oracle (C5); numeric threshold required for other sources
       seed: int,
   ) -> dict[str, Any]:
       """Returns per-cell diagnostics including condition, priors_source, metrics, predicted_edges, status, error, constraint_summary, dropped_due_to_cycle (GES)."""
   ```
   - Strict validation: `priors is None` and `threshold is None` only for C0 (`priors_source="none"`); `priors_source="oracle"` builds claims from `true_graph`; all other sources require non-`None` `priors` and `threshold`.
   - If `priors is None` → no constraints (C0 baseline).
   - Otherwise build constraints via `ConstraintBuilder` at the given threshold.
   - Set the seed before running the algorithm.
   - For PC/LiNGAM: pass constraints natively.
   - For GES: run unconstrained GES, then apply post-hoc edits.
   - Compute metrics via `evaluation/harness.py`.
   - Return one dict.

5. **Persist results incrementally** to `experiments/discovery_results_sachs.json`:
   ```json
   {
     "dataset": "sachs",
     "results": [
       {
         "condition": "C2",
         "algorithm": "LiNGAM",
         "seed": 0,
         "threshold": 0.7,
         "priors_source": "reactome_llm",
         "metrics": {"shd": 14.0, "aupr": 0.51, "precision": 0.62, "recall": 0.41, "f1": 0.49},
         "predicted_edges": [["praf", "pmek"], ...],
         "constraint_summary": {"hard_required": 5, "soft_prior": 11, "discarded": 28}
       }
     ]
   }
   ```
   - Append-and-flush after every cell so a crash mid-sweep is recoverable.
   - Make the runner idempotent: skip cells whose `(condition, algorithm, seed, threshold)` is already in the file.

6. **Wire the runner.** `experiments/runner.py` will be implemented in Step 6, but expose `run_condition` as the unit it calls.

---

## Acceptance criteria

- [ ] `ConstraintBuilder.summary()` correctly accounts for every prior at every threshold (no priors lost).
- [ ] PC, GES, and LiNGAM all run cleanly under C0, C0.5, C1, C2, C3, C4 conditions on Sachs (these conditions defined in `dev_plan_v2.md` §3 Step 6 and Step 4 deliverables).
- [ ] Predicted graphs serialise to both edge lists and adjacency matrices in the JSON output.
- [ ] Re-running the runner skips already-completed cells (idempotency).
- [ ] `task lint` and `task test` pass.

---

## Pitfalls

- **LiNGAM prior-matrix convention.** Default fill = `-1`, required = `1`, forbidden = `0`. Match `lingam.DirectLiNGAM`'s `prior_knowledge` API. The current codebase is already correct; don't change it.
- **`BackgroundKnowledge` requires `GraphNode` instances.** `add_required_by_node(GraphNode(name_a), GraphNode(name_b))` — strings will silently fail.
- **GES caveat.** Post-hoc edits *can* introduce cycles. Ensure `apply_constraints` either (a) refuses to add an edge that creates a cycle, or (b) the runner records when this happens. Don't silently produce non-DAGs.
- **Threshold semantics.** `confidence ≥ threshold` keeps the edge. Be explicit: `≥` not `>`. Off-by-one will quietly change conditions.
- **Seed reproducibility.** PC's `causal-learn` implementation uses `numpy.random` for some test order tie-breaking. Set the seed via `np.random.seed(seed)` before calling. LiNGAM's `DirectLiNGAM` accepts `random_state` directly.
- **Conflict between priors and data.** If the LLM emits a `hard_required` edge that the data strongly contradicts, PC may fail to construct a valid CPDAG. Catch the exception, log it, and record the cell as `failed` rather than crashing the sweep.

---

## Out of scope

- The full sweep across all conditions × seeds (Step 6 orchestrates).
- Reporting / figures (Step 6).
- DREAM4 (Step 7).
- pytetrad fGES (parking lot).
