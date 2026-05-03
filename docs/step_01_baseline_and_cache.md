# Step 1 — Sachs Baseline + Lightweight Cache

**Role.** Foundation. Critical-path start.

**Goal.** Reproduce the unconstrained PC / GES / LiNGAM baseline on the Sachs dataset, and build the lightweight on-disk response cache used by both the LLM client and the Reactome REST client in later steps.

**Depends on.** Nothing.

**Effort.** 1–1.5 days. The cache itself is ~30 lines.

---

## Deliverables

| Artefact | Purpose |
|----------|---------|
| `llm/cache.py` | Tiny on-disk JSON cache, keyed by SHA256 of the request payload; reusable for any HTTP/LLM client |
| `llm/client.py` | OpenRouter LLM client wrapper that consults the cache before hitting the API; records `served_model` from each response |
| `evaluation/harness.py` | `evaluate(predicted, true) -> {metric: value}` |
| `experiments/baseline_sachs.py` | Runner script |
| `experiments/baseline_sachs.json` | Mean ± std of SHD, AUPR, Precision, Recall, F1 across 10 seeds × 3 algorithms |
| Tests in `tests/test_cache.py` and `tests/test_harness.py` | Coverage for the cache + harness |

---

## Implementation tasks

1. **Add dependencies to `pyproject.toml`.**
   - `httpx` (used by the Reactome REST client in Step 2 and `llm/client.py` if migrating off the OpenAI SDK; add now to centralise `task install`).
   - `pypath-omnipath` (used in Step 4 — same reason).
   - Verify `cdt` works for `load_dataset('sachs')` without R/Java extras.

2. **Implement `llm/cache.py` (the lightweight cache).**
   ```python
   import hashlib, json
   from pathlib import Path
   from typing import Any

   def cache_key(payload: dict[str, Any]) -> str:
       return hashlib.sha256(
           json.dumps(payload, sort_keys=True).encode("utf-8")
       ).hexdigest()

   def cached_call(cache_dir: Path, payload: dict, fetch):
       cache_dir.mkdir(parents=True, exist_ok=True)
       key = cache_key(payload)
       path = cache_dir / f"{key}.json"
       if path.exists():
           return json.loads(path.read_text())
       result = fetch()
       path.write_text(json.dumps(result, default=str))
       return result
   ```
   - One file per key under `cache/llm/` (and later `cache/reactome/`).
   - Same helper reused by the Reactome client in Step 2.
   - That's it. No sharding, no atomic-write rename, no stage tags. Total: ~20 lines.

3. **Update `llm/client.py` to use the cache.**
   - `MODEL` is pinned to `nvidia/nemotron-3-super-120b-a12b:free` (changed from `openrouter/free` after Step 3 found the alias routed to a 1.2B model that could not ground biomedical column names; see dev plan changelog item 5 and §8 for the full model-selection trail).
   - Wrap `client.chat.completions.create(...)` with `cached_call(Path("cache/llm"), {"model": model, "messages": messages, **kwargs}, fetch)`.
   - Store the *full* response object so token counts and `served_model` from OpenRouter survive replay.
   - Extract `served_model = response.model` (the underlying provider OpenRouter routed to) and persist it alongside each cache entry — this is what the paper appendix reports.

4. **Implement `evaluation/harness.py`.**
   - `evaluate(predicted: nx.DiGraph, true: nx.DiGraph) -> dict[str, float]` returning `{shd, aupr, precision, recall, f1}`.
   - Reuse `evaluation/metrics.py` primitives.
   - Make sure both graphs have the same node set before metric computation (add missing isolated nodes).

5. **Implement `experiments/baseline_sachs.py`.**
   - `from cdt.data import load_dataset; data, true_graph = load_dataset('sachs')`.
   - Run PC (α=0.05, fisherz), GES (BIC), DirectLiNGAM with default hyperparams.
   - 10 seeds (0–9). For algorithms that don't accept a seed (PC), seed any data subsampling/shuffling explicitly.
   - Compute mean ± std per metric.
   - Write `experiments/baseline_sachs.json`:
     ```json
     {
       "PC": {"shd": {"mean": 18.4, "std": 0.0}, ...},
       "GES": {...},
       "LiNGAM": {...}
     }
     ```
   - Print a formatted summary table to stdout.

6. **Add `task baseline-sachs` to `Taskfile.yml`** that runs the baseline script.

---

## Acceptance criteria

- [ ] PC SHD on Sachs lands in published range (~17–22, depending on metric variant).
- [ ] Re-running any LLM-calling code path with the network disabled succeeds and returns identical results from cache.
- [ ] Each cached LLM response includes the `served_model` from OpenRouter so the appendix can report the underlying providers.
- [ ] `task test` passes; tests cover: (a) cache hit / miss / replay, (b) cache key determinism across runs, (c) harness on a tiny synthetic graph pair.
- [ ] `task lint` passes.
- [ ] `experiments/baseline_sachs.json` is committed.

---

## Pitfalls

- **LiNGAM prior-matrix convention.** The current codebase uses `-1` as the default fill, `1` for required, `0` for forbidden. Keep this — it matches `lingam`'s `DirectLiNGAM(prior_knowledge=...)` API. Don't follow v1's `0`-as-default convention.
- **`cdt` Java/R extras.** `cdt` drags Java/R deps for some causal discovery methods, but `load_dataset('sachs')` and the metric helpers (`SHD`, `precision_recall`) work without them. Don't accidentally import an R-backed method.
- **Cache key sensitivity.** Include `temperature`, `top_p`, and any other generation kwargs in the cache key payload. Otherwise a temperature change silently reuses old responses.
- **`openrouter/free` is a routing alias.** Two requests with the same prompt may be served by *different* underlying models on different days. The cache pins each call's response so this doesn't break paper numbers — but record `served_model` so the appendix can report which provider served what. Without the cache, results would not be reproducible.
- **Streaming responses.** If the OpenAI client streams, materialise to a complete response before caching. Streamed objects don't serialise well.

---

## Out of scope

- Variable grounding (Step 3).
- Reactome client (Step 2).
- Any non-Sachs dataset.
