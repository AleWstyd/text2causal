# Reactome Content Service — Endpoints used by `reactome.client`

Base URL: `https://reactome.org/ContentService`

The client only needs five endpoints. Every response is cached on disk under
`cache/reactome/` keyed by `sha256(json.dumps({"path": ..., "params": ...}, sort_keys=True))`,
so reruns with the network disabled succeed when every requested payload
already has a cache file.

| # | Endpoint | Purpose | Verified example |
|---|----------|---------|------------------|
| 1 | `GET /data/mapping/UniProt/{accession}/reactions` | Protein → reactions involving it directly or via Complex membership. Returns inferred reactions from other species too — re-filter to `speciesName == "Homo sapiens"`. | `RAF1` (`P04049`) → 64 human reactions |
| 2 | `GET /search/query?query={name}&species=Homo%20sapiens&types=Chemical%20Compound` | Metabolite name → top SimpleEntity stId. `name` field comes back HTML-highlighted (`<span class="highlighting">…</span>`); strip before display. | `phosphatidylinositol-4,5-bisphosphate` → SimpleEntity stId for PIP2 |
| 3 | `GET /data/query/enhanced/{stId}` | Full reaction or entity object with `input` / `output` / `catalystActivity` / `regulatedBy` returned inline (and `consumedByEvent` / `producedByEvent` for SimpleEntities). | `R-HSA-5672978` ("RAF phosphorylates MAP2K dimer") → 3 inputs, 3 outputs, 1 catalyst, 2 regulators |
| 4 | `GET /data/participants/{stId}` | Per-PhysicalEntity flattened reference entities for a reaction. Returns one entry per PE with `peDbId` and a `refEntities` list expanded through Complex / EntitySet membership. | The RAF/MEK/MAPK complex returns 41 expanded refs incl. P04049 (RAF1), Q02750 (MAP2K1), P36507 (MAP2K2) |
| 5 | `GET /data/discover/{stId}` | Schema.org metadata (name, description, citations) for reactions, complexes, or pathways — used by `format_context_for_llm` paragraph generation when extra prose is wanted. | Useful as a fallback display name source |

## Pitfalls reflected in the client

- **`input` / `output` / `regulatedBy` mix dicts and primitive ints.** The
  client filters `if isinstance(x, dict)` before processing and treats bare
  integers in `regulatedBy` as a separate `query/enhanced/{dbId}` follow-up.
- **Catalysts / regulators are often Complexes, not bare proteins.** The
  `participants` endpoint expansion handles this: a Complex's `refEntities`
  lists every UniProt inside it.
- **Search results are HTML-highlighted.** The client strips
  `<span class="highlighting">…</span>` from any user-visible name.
- **ChEBI ids are inconsistent.** The client normalises to `CHEBI:<digits>`.
- **`/data/mapping/ChEBI/{id}/reactions` is unreliable** (404 for PIP2 in
  testing). The client uses search-then-`/data/query/enhanced` for
  metabolites instead.
- **Inferred-from-other-species reactions** are silently filtered out by
  matching `speciesName == "Homo sapiens"`.

## Fallback (out of scope)

If a future Reactome change makes REST insufficient, the documented fallback
is **native Neo4j Community Edition via Homebrew** plus the Reactome data
dump (`brew install neo4j` → load dump → run on `localhost:7687`). This
preserves the original Cypher path at the cost of ~2 hours of setup.
