# Step 2 — Reactome REST Client and Coverage

**Role.** Foundation.

**Goal.** Build a thin Python client over the Reactome **Content Service REST API** (`https://reactome.org/ContentService`), with on-disk caching, sufficient to retrieve role-aware reaction context for any pair of grounded variables. Verify coverage on Sachs and DREAM4 PSN before committing to downstream steps.

**Depends on.** Step 1 (need dataset variable names for the coverage check; reuse the `cached_call` helper).

**Effort.** 2–3 days. (Down from 3–4 in v1 because the Day-1 schema spelunking has already been done — see "Verified endpoints" below — and there is no Docker / Neo4j to set up.)

---

## Why REST and not Cypher / Docker

- The user has no Docker available locally.
- The Reactome Content Service is free, public, no-auth, and exposes everything we need.
- Two endpoints (`/data/query/enhanced/{stId}` + `/data/participants/{stId}`) jointly return role-typed participants with Complex membership flattened into UniProt / ChEBI references — no client-side recursion.
- The same on-disk cache pattern from Step 1 wraps Reactome calls; reruns are offline.

If a future Reactome change makes REST insufficient, the documented fallback is **native Neo4j Community Edition via Homebrew** plus the Reactome data dump (`brew install neo4j` → load dump → run on `localhost:7687`). This is more setup but preserves the original Cypher path.

---

## Deliverables

| Artefact | Purpose |
|----------|---------|
| `reactome/client.py` | `ReactomeClient`, `EntityRef`, `ReactionRecord`, `format_context_for_llm` |
| `reactome/endpoints.md` | Reference list of REST endpoints and example responses (verified) |
| `experiments/reactome_coverage.json` | Coverage report for Sachs and (preliminarily) DREAM4 PSN |
| Tests in `tests/test_reactome_client.py` | Cache hits, role parsing, family expansion |

---

## Verified endpoints (Day-1 spelunking already done)

These have been confirmed to return useful data on real reactions; they form the API surface our client uses.

| Endpoint | Purpose | Example |
|---|---|---|
| `GET /search/query?query={name}&species=Homo%20sapiens&types={Protein\|Chemical%20Compound}` | Name → stId search (used for ChEBI metabolites that don't have a direct mapping endpoint) | `query=phosphatidylinositol-4,5-bisphosphate` → `R-HSA-177939` (PI3K converts PIP2 to PIP3) |
| `GET /data/mapping/UniProt/{accession}/reactions` | UniProt → list of reactions involving that protein (directly or via Complex membership) | RAF1 (P04049) → 64 human reactions |
| `GET /data/query/enhanced/{rxnStId}` | Full reaction object with `input` / `output` / `catalystActivity` / `regulatedBy` fields. Catalyst's `physicalEntity` is dereferenced inline. Regulations are returned as `NegativeRegulation` / `PositiveRegulation` objects with the regulator entity inline. | R-HSA-5672978 ("RAF phosphorylates MAP2K dimer") returns 3 inputs, 3 outputs, 1 catalyst, 2 regulators |
| `GET /data/participants/{rxnStId}` | Per-PhysicalEntity flattened references — for each PE in the reaction, returns its `refEntities` list with all UniProt / ChEBI accessions visible through Complex / EntitySet membership | The RAF/MEK/MAPK complex returns 41 expanded refs incl. P04049 (RAF1), Q02750 (MAP2K1), P36507 (MAP2K2) |
| `GET /data/discover/{stId}` | Schema.org metadata for any entity (name, description, citations) | Useful for `format_context_for_llm` paragraph generation |

**Caveat:** `/data/mapping/ChEBI/{id}/reactions` returned 404 for PIP2 (CHEBI:18348). Workaround: for metabolites, search by canonical name first to get the relevant SimpleEntity stId, then traverse via `componentOf` / `consumedByEvent` / `producedByEvent` fields on the entity. Sachs has only PIP2 + PIP3 in this category, so this is two extra search steps total.

---

## Implementation tasks

### Day 1 — REST client + cache integration (~half day)

1. **Implement `reactome/client.py`** as a thin wrapper around `httpx`:
   ```python
   import httpx
   from pathlib import Path
   from llm.cache import cached_call

   REACTOME_BASE = "https://reactome.org/ContentService"
   CACHE_DIR = Path("cache/reactome")

   def _get(path: str, **params) -> dict | list:
       def fetch():
           r = httpx.get(f"{REACTOME_BASE}{path}", params=params, timeout=30.0)
           r.raise_for_status()
           return r.json()
       return cached_call(CACHE_DIR, {"path": path, "params": params}, fetch)
   ```
   - Use the same `cached_call` helper from Step 1.
   - Add a `Retry-After`-aware retry-with-backoff for transient 429 / 5xx (3 attempts, exponential backoff).

2. **Define the typed wrappers:**
   ```python
   @dataclass(frozen=True)
   class EntityRef:
       kind: Literal["protein", "metabolite"]
       ids: list[str]              # UniProt or ChEBI accessions
       display_name: str

   @dataclass(frozen=True)
   class ReactionRecord:
       reaction_id: str            # R-HSA-...
       reaction_name: str
       pathway: str | None
       role_a: Literal["input", "output", "catalyst", "regulator"]
       role_b: Literal["input", "output", "catalyst", "regulator"]
       reaction_type: str          # "Reaction", "BlackBoxEvent", etc.
       sign: Literal["positive", "negative", "unknown"] | None  # only meaningful when role is regulator
   ```

### Day 1 (afternoon) — Core query functions

3. **`reactions_for_protein(uniprot: str) -> list[ReactionMeta]`**
   - Hit `/data/mapping/UniProt/{uniprot}/reactions`.
   - Filter to `speciesName == "Homo sapiens"` (the API often returns inferred reactions from other species; explicit filter is safer than a query param).

4. **`reactions_for_metabolite(name: str) -> list[ReactionMeta]`**
   - Hit `/search/query?query={name}&species=Homo%20sapiens&types=Chemical%20Compound`.
   - Take top hit's stId (a SimpleEntity), then call `/data/query/enhanced/{stId}` and read its `consumedByEvent` and `producedByEvent` fields to find reactions.
   - Used only for the 2 Sachs metabolites (PIP2, PIP3).

5. **`reaction_details(rxn_stId: str) -> dict`**
   - Hit `/data/query/enhanced/{rxn_stId}`.
   - Returns the full reaction object — caller extracts `input`, `output`, `catalystActivity`, `regulatedBy`.

6. **`reaction_participants(rxn_stId: str) -> dict[int, list[ReferenceEntity]]`**
   - Hit `/data/participants/{rxn_stId}`.
   - Returns a mapping `peDbId → [ReferenceEntity]` so the caller can look up "for the catalyst PhysicalEntity (peDbId=X), which UniProts/ChEBIs are in its expansion?"

### Day 2 — Co-participation and role assignment

7. **`get_reaction_context(a: EntityRef, b: EntityRef) -> list[ReactionRecord]`**
   - Get reaction lists for `a` and `b` via the appropriate per-kind function, OR over family ID lists.
   - Intersect by `stId` to find shared reactions.
   - For each shared reaction, call `reaction_details` *and* `reaction_participants`.
   - For each participant role (`input` / `output` / `catalystActivity` / `regulatedBy`), find which PhysicalEntity stIds map to entity `a` and which to entity `b`, by joining `query/enhanced`'s PE stIds against `participants`' refEntities (UniProt / ChEBI).
   - Emit a `ReactionRecord` per (reaction, role-of-a, role-of-b) combination. (A single reaction can contribute multiple records if e.g. RAF1 is in both the catalyst Complex and an input.)

8. **`format_context_for_llm(records, name_a, name_b) -> str`**
   - Output a clean natural-language paragraph per reaction record. Example:
     > "In the RAF/MAP kinase cascade pathway, RAF1 acts as a member of the catalyst complex in reaction R-HSA-5672978 ('RAF phosphorylates MAP2K dimer'), where MAP2K1 is a member of the output complex. Reaction type: Reaction."

### Day 2 (afternoon) / Day 3 — Coverage check

9. **Implement `experiments/reactome_coverage.py`:**
   - For each Sachs node (using a temporary hand-coded mapping until Step 3): does `reactions_for_protein` (or `_metabolite`) return ≥ 1 hit?
   - For each Sachs node *pair*: does `get_reaction_context` return ≥ 1 record?
   - Compute and write:
     ```json
     {
       "dataset": "sachs",
       "node_coverage": "11/11",
       "pair_coverage": "X/55",
       "mean_reactions_per_node": 18.7,
       "mean_records_per_pair_with_context": 4.3,
       "nodes_missing": [],
       "pairs_missing": [...],
       "served_endpoints_total": 1437,
       "served_from_cache_on_rerun": "100%"
     }
     ```

### Day 3 — Buffer + tests

10. Tests for `ReactomeClient` against cached fixtures (committed under `tests/fixtures/reactome/`).
11. Add `task reactome-coverage` to `Taskfile.yml`.

---

## Acceptance criteria

- [ ] `get_reaction_context(EntityRef("protein", ["P04049"], "RAF1"), EntityRef("protein", ["Q02750"], "MAP2K1"))` returns at least one `ReactionRecord` for `R-HSA-5672978` with `role_a="catalyst"` and `role_b="output"`.
- [ ] `get_reaction_context(EntityRef("metabolite", ["CHEBI:18348"], "PIP2"), ...)` returns at least one record (verifies the metabolite path works).
- [ ] Sachs `node_coverage` = 11/11 (hard).
- [ ] Sachs reaction-level `pair_coverage` recorded in `experiments/reactome_coverage.json`. The `≥ 0.5` line is **aspirational only** — escalation is keyed to `node_coverage < 0.8` per the dev-plan Risk Register, not to pair coverage. If reaction-level pair coverage is < 0.5, Step 2.5 (pathway / regulator-chain / co-complex layers) closes the gap before Step 4; OmniPath-as-primary is not triggered.
- [ ] Re-running the coverage script with the network disabled succeeds (cache replay).
- [ ] `task lint` and `task test` pass.

---

## Tripwire (escalation)

If `node_coverage < 0.8` on the primary dataset (extremely unlikely for Sachs given the verified RAF1 result):
- Investigate whether the gap is a grounding problem (Step 3 might fix it) or a true Reactome gap.
- If true Reactome gap, escalate to OmniPath as the primary structured source for that dataset and document in `experiments/reactome_coverage.json`.
- Native Neo4j via Homebrew is the second fallback if both REST and OmniPath are insufficient — full Cypher access at the cost of ~2 hours of setup (install Neo4j, download Reactome graph dump, load).

---

## Pitfalls

- **`input` / `output` / `regulatedBy` fields can mix dicts and primitive ints.** The dicts are real PhysicalEntity records; the primitive ints are stoichiometry indices or shallow dbId references. Filter `if isinstance(x, dict)` before processing.
- **`regulatedBy` may contain shallow references that need a follow-up call.** In testing, one entry was a full Regulation dict (with `regulator` inline) and another was a bare integer. Treat bare integers as a separate `data/query/enhanced/{dbId}` call.
- **Catalysts and regulators are often Complexes, not bare proteins.** This is expected — Reactome models signaling proteins as members of scaffolded complexes. The `participants` endpoint expansion handles this; a Complex's `refEntities` lists every UniProt inside it.
- **Search results are HTML-highlighted.** Names in the search response come back wrapped in `<span class="highlighting">...</span>`. Strip these when displaying.
- **Rate limits.** The Content Service is generally generous but undocumented for free use. The cache makes most reruns offline; for first-time sweeps add a polite `httpx.AsyncClient` with `limits=httpx.Limits(max_connections=10)` if you parallelise.
- **ChEBI ID format.** Reactome stores some ChEBI references with `CHEBI:` prefix, others as bare numbers. Normalise in the client.
- **Inferred-from-other-species reactions.** UniProt-mapping responses sometimes include orthologous reactions from rat/mouse/etc. Always re-filter by `speciesName == "Homo sapiens"`.
- **Curation modality vs coverage gap.** Reactome curates PKC- and PKA-mediated regulation of MAPK substrates at *pathway* scope (e.g. "Activation of Protein Kinase C", "PKA-mediated phosphorylation of CREB and other targets"), not as one reaction per substrate. Empirically, `reactions_for_protein("P17252")` (PKC-α) and `reactions_for_protein("P04049")` (RAF1) have an empty `stId` intersection even after expanding to full PKC and PKA pan-isoform UniProt sets. This is a curation property of Reactome, **not** a grounding bug or a client bug. The fix is Step 2.5's pathway / regulator-chain / co-complex expansion layers, not OmniPath escalation.

---

## Out of scope

- Variable grounding (Step 3 will replace the temporary hand-coded mapping used in coverage).
- LLM calls of any kind.
- OmniPath integration (Step 4).
