# Research Development Plan v2

**Knowledge-Grounded LLM Agent for Causal Discovery Using Reactome and the Sachs Protein-Signalling Dataset**

Mikołaj Jarosławski, Dominik Sepioło, Antoni Ligęza
AGH University of Krakow — Department of Applied Computer Science
Plan version: 2 — May 2026

---

## Changelog vs v1

This document supersedes `CausalDiscovery_DevPlan.pdf` (v1). Material changes:

1. **Reactome access pattern changed.** v1 assumed local Neo4j via Docker plus hand-written Cypher; v2 uses the public Reactome Content Service REST API. The schema corrections discovered in v1 review (catalystActivity / regulation as 2-hop paths via `CatalystActivity` / `Regulation` nodes; species via a `Species` node, not a relationship property) are still relevant — they describe the underlying graph the REST API exposes — but the REST `query/enhanced` endpoint flattens them inline so client code never has to traverse those hops manually.
2. **Variable grounding generalised** to support `uniprot_id | chebi_id | list[uniprot_id]`. PIP2 and PIP3 are phosphoinositide metabolites (ChEBI), not proteins. PKC, PKA, ERK1/2 are protein families that map to multiple UniProt IDs.
3. **OmniPath added as a secondary structured-knowledge source** for the no-LLM floor ablation (C0.5). Cheap to integrate, preempts the "is the gain Reactome-specific or just any structured ontology?" reviewer objection.
4. **Two new ablation conditions added** (C0.5: Reactome-only no-LLM; C1: free-text-LLM with current skeleton). Together they isolate the marginal contribution of (a) the ontology over no-knowledge, (b) the LLM over the ontology, and (c) the ontology over plain free-text-LLM.
5. **Lightweight on-disk caching for LLM and Reactome calls.** Single JSON-per-key under `cache/llm/` and `cache/reactome/`, ~30 lines of code total. Same hash-based key pattern for both. Caches committed to the repo so the pipeline can replay without API access. The model is pinned to `Qwen/Qwen3-235B-A22B` served from a Baseten dedicated deployment via an OpenAI-compatible API; `BASETEN_API_KEY` is the live-call credential. The Step 3 model-selection trail is documented for the paper appendix: the original `openrouter/free` routing alias was rejected after it routed requests to a 1.2B-parameter `liquid/lfm-2.5-1.2b-thinking:free` model that hallucinated identifiers; OpenRouter free-tier alternatives (`meta-llama/llama-3.3-70b-instruct:free`, `qwen/qwen3-next-80b-a3b-instruct:free`, `nousresearch/hermes-3-llama-3.1-405b:free`) were upstream rate-limited; `nvidia/nemotron-3-super-120b-a12b:free` and `openai/gpt-oss-120b:free` were operational but exhibited the same failure mode that ultimately drove the Step 3 architecture: open-weight models in the 100B-235B class know gene symbols reliably but mis-recall specific UniProt accessions for less-common kinase isoforms. Pinning Qwen3-235B-A22B on Baseten gives a stable open-weight endpoint; the gene→UniProt resolution step (Step 3 §6, below) makes Step 3 quality independent of which open-weight model is used. Each cached LLM response still records the actually-served model so the paper appendix can document it.
6. **Secondary dataset is DREAM4 Predictive Signalling** (MCF7 breast cancer, ~7 antibody-measured human proteins), not the DREAM4 Network Inference subset. The signalling subset is Reactome-coverable and biologically a sibling of Sachs; the network-inference subset is *E. coli* / *S. cerevisiae* and would not exercise Reactome.
7. **Effort estimates revised upward** from 0.5–1 day per step to realistic 6–8 weeks total focused work.
8. **Venue retargeted to CLeaR 2027 (primary) or ECAI 2027 (fallback).** ECAI 2026 deadline has likely passed; RAI 2026 is feasible only with an aggressive June/July submission window.
9. **GES post-hoc constraint application acknowledged as a methodological caveat**, not glossed over. To be reported clearly in the paper or replaced.
10. **Decisions deferred:** fate of the existing Streamlit UI (`main.py`); deterministic rule-based Step-4 reasoner (currently out of scope, may be added as a stretch goal).
11. **LLM-only baseline added (`C-LLM-only`).** The cached LLM causal claims are also emitted directly as a predicted DAG (no causal-discovery algorithm), and run alongside the other conditions. This empirically tests whether the CD step adds value beyond LLM judgments — addresses the Kıcıman et al. (2023) line of work in which LLMs alone are competitive on Sachs pairwise. Cost is near-zero because the claims already exist as a Step 4 output.
12. **Reactome accessed via the Content Service REST API (no Docker).** Verified to expose the four role types we need (`input`, `output`, `catalystActivity`, `regulatedBy`) via the `/data/query/enhanced/{stId}` endpoint, with Complex membership flattened by `/data/participants/{stId}` (UniProt and ChEBI references inlined). Two HTTP calls per reaction; cached on disk. Replaces the `docker-compose.yml` plus Neo4j Bolt setup from v1. Native Neo4j via Homebrew is documented as a fallback only if a future API change makes REST insufficient.
13. **Step 4 shipped: causal reasoning + OmniPath floor + LLM-only DAG.** Added `reasoning/reason.py` (LLM directional claims with conflict-aware aggregation), `reasoning/dag_from_priors.py` (cycle-broken C-LLM-only DAG), and `omnipath_floor/floor.py` (the spec calls for `omnipath/`, but the PyPI `omnipath` REST client occupies that import path; renamed locally to `omnipath_floor/`). LLM transport was swapped mid-step from the retired `Qwen/Qwen3-235B-A22B` Baseten dedicated deployment to `deepseek-ai/DeepSeek-V4-Pro` on the Baseten Model API; both `LLM_BASE_URL` and `LLM_MODEL` remain env-overridable. The model-selection trail is fully documented (item 5, plus `docs/step_04_causal_reasoning.md` "LLM transport"). Sachs full-sweep result: 110 ordered pairs, 62 with Reactome context, 16 high-confidence claims, 0 conflicts, 122 LLM calls, all served by DeepSeek-V4-Pro; 100% cache-hit invariant verified offline. The C-LLM-only DAG retains 16 biologically sensible edges (MAPK cascade + PI3K/Akt branch + PKA→RAF) and drops 2 cycle-closing edges. OmniPath floor: 43 directed edges with `source_filter='all'`, 3 with `source_filter='reactome_only'`. Two new ablation conditions (C0.5 + C-LLM-only) are now ready for Step 5/6 integration. Pypath (`pypath-omnipath`) is currently broken upstream (`pysftp`/`paramiko` `DSSKey` import error) — the lighter `omnipath` package replaces it for this work.
14. **Step 6.5 stabilisation completed.** Sparse LiNGAM priors were tested (`sparse_required`, `forbidden_only`, `hybrid_top5`); the best successful Reactome+LLM LiNGAM cell was C3 `forbidden_only` with F1=0.566, below C0 LiNGAM F1=0.593. PC priors change edge sets and improve AUPR (0.52→0.56) but not F1. The paper framing shifts to constraint quality and auditability: on Reactome-covered Sachs pairs, Reactome+LLM has precision=0.53, recall=0.69, strict hallucination=0.00 vs OmniPath-all precision=0.17, recall=0.31, strict hallucination=0.26. See `docs/step_06_5_stabilisation.md`.
15. **Step 7 tripwire fired.** The official DREAM4 Predictive Signalling Network Modeling Synapse project (`syn2825304`) exists, but file access/listing requires authenticated Synapse access in this environment and no public mirror with CSV + ground truth was available. The documented fallback was used: a 7-node synthetic human-signalling SCM derived from the DREAM4 HepG2 phosphoprotein panel and Reactome/literature-compatible pathway structure. Artefacts live under `data/dream4_psn/`, `experiments/*dream4_psn*`, `tables/ablation_table_dream4.tex`, `tables/cross_dataset.tex`, and `figures/gap_closed_dream4.{pdf,png}`.
16. **Directed vs skeleton F1.** The existing `precision`/`recall`/`f1` keys in `evaluate()` and run JSONs remain **skeleton** metrics (undirected edge overlap; reversed arcs count as correct). New keys `directed_precision`, `directed_recall`, and `directed_f1` compare ordered arcs. Step 6 reporting uses **directed F1** as the headline metric (`tables/ablation_table_directed.tex`, gap-closed figures, headline summary); `tables/ablation_table.tex` is retained for skeleton-F1 provenance. `tables/aupr_extension.tex` adds an F1\textsubscript{dir} column. Older committed result JSONs are backfilled with directed keys when `experiments.runner` / `experiments.run_discovery_sachs` detect their absence (no full sweep rerun).
17. **C3+ft free-text fallback for the Reactome no-context stratum.** On Sachs, `experiments/constraint_quality_sachs.json` shows 22 unordered true-graph pairs sit in the union `no_coverage` stratum (5 of 18 true edges), while the cached paragraph free-text prior (`experiments/freetext_priors_sachs.json`) emits 6 edges at precision 1.00 against gold. `reasoning/merge_freetext_fallback.py` merges those claims into `causal_priors_*_with_fallback.json` only where the Reactome pipeline emitted `no_context`, setting per-claim `source=freetext_fallback` (or `per_pair_freetext_fallback` for PR2b). Step 6 adds condition **C3+ft** (same τ=0.7 as C3) loading the merged priors; Reactome-grounded rows are unchanged. **Empirical note (pre-PR2b):** the 6 committed paragraph free-text edges concerned pairs that already appear in `pairs` with Reactome context, not in `no_context_pairs`, so `n_fallback_applied` was 0 on Sachs and C3+ft matched C3 until PR2b’s per-pair pass filled the gap (changelog item 20).
18. **PC post-hoc required edges + LiNGAM `forbidden_only` default (PR3).** PC native `BackgroundKnowledge` does not stop Fisher-Z from removing required skeleton edges; the pipeline injects required edges after PC with `apply_post_hoc_edits`, logging cycle drops in `dropped_due_to_cycle` / `pc_post_hoc_*` (same structural caveat as GES post-hoc). For constrained conditions, canonical LiNGAM uses a soft `forbidden_only` sparse prior matrix; dense required (`build_lingam_prior_matrix`) remains available via `lingam_prior_mode="all"` for sweeps.
19. **LiNGAM post-hoc constraint injection (PR3b).** The PR3 `forbidden_only` soft mode returned directed F1=0 on C5 oracle because `apply_prior_knowledge_softly=True` does not actually block reverse edges in the output graph — DirectLiNGAM's coefficient threshold (0.001) leaves offending arcs intact. For the canonical Sachs matrix, constrained LiNGAM now runs **unconstrained** first and then applies the same `apply_post_hoc_edits` required-add / forbidden-remove pass as GES and PC (`lingam_prior_mode="post_hoc"`). This unifies how all three algorithms treat priors (all three are "post-hoc with cycle-drop logging"); aligns the dev-plan Risk #6 caveat across the suite instead of only GES + PC; and moves C5 oracle LiNGAM directed F1 from 0.00 → 0.48 and SHD from 46 → 26, giving a proper dose-response (C0 0.37 < C3 0.40 < C5 0.48). The three native-LiNGAM encodings (`all`, `sparse_required`, `forbidden_only`, `hybrid_top5`) remain selectable for the existing sweep scripts.
20. **Per-pair parametric LLM fallback for Reactome no-context pairs (PR2b).** Paragraph free-text priors rarely overlap `no_context_pairs`, so **C3+ft** previously matched **C3** on Sachs. `reasoning/freetext_fallback.py` adds one cached Baseten call per *unordered* `no_context` pair (no Reactome context in the prompt), merged after paragraph freetext in `task fallback-priors`. Sachs (May 2026 run): 24 LLM calls → 18 directional claims kept (6 pairs skipped after validation); merging into `causal_priors_sachs_with_fallback.json` applies **36** slot substitutions because the priors artefact lists both ordered endpoints for each gap. DREAM4 PSN’s committed `causal_priors_dream4_psn.json` has **no** `no_context_pairs` (0 calls). LiverDREAM: 13 calls, 8 claims kept, **16** merge slots. **Sachs downstream vs C3 (mean over 10 seeds):** PC directed F1 unchanged at **0.524**, SHD **20**; **GES** directed F1 **0.491 → 0.546** (+0.055), SHD **27 → 25**; **LiNGAM** **0.400 → 0.453** (+0.053), SHD **33 → 29** — GES/LiNGAM headline cells move to **C3+ft** in Step 6 reporting. All **6** Sachs true edges whose unordered endpoints lie in the Reactome no-context stratum (PKA→{pjnk, p44/42, pmek}, PKC→{pjnk, pmek, praf}) match the per-pair artefact’s oriented claims. `constraint_quality_sachs.json`: `reactome_llm_with_freetext_fallback` recall **0.83** vs **0.50** for context-only Reactome+LLM at the same τ.

---

## 1. Scientific Background and Motivation

### 1.1 Research lineage

The project sits at the convergence of three research lines from AGH:

- **Model-Driven XAI (Sepioło & Ligęza, *Applied Sciences* 2024).** The 3C principle — Causality, Components, Connections — frames causal structure discovery as a prerequisite for white-box, model-driven explainability. Shallow post-hoc methods (LIME, SHAP) cannot incorporate external declarative knowledge; the alternative is to derive interpretable models guided by domain knowledge.
- **Causal Discovery Benchmarking (Jarosławski, Sepioło & Ligęza, *RAI* 2025).** Empirical evaluation of PC, GES, and LiNGAM on the LUCAS dataset using `causal-learn`. Domain-knowledge constraints improve LiNGAM meaningfully (AUPR 0.312 → 0.495; Recall 0.33 → 0.58); PC and GES are less sensitive. Establishes the metric stack (AUPR, SHD, Precision, Recall, F1) and ablation protocol inherited here.
- **LLM-Augmented Evolutionary Computation (Sepioło, Jarosławski & Ligęza, *RAI* 2025).** LLMs can iteratively refine grammars for Grammatical Evolution. Demonstrates both the potential and the current limits of LLM-in-the-loop scientific workflows: partial human intervention was still needed for correct solutions.

### 1.2 Related work

| System / Paper | Knowledge source | Mechanism | Gap vs. this work |
|---|---|---|---|
| CausalCopilot (Wang et al., 2025) | Implicit (LLM parametric) | End-to-end automation: algorithm selection, hyperparameter tuning, post-processing with LLM pruning and direction revision | Knowledge is implicit and non-auditable; no structured ontology; no variable grounding |
| ALCM (Khatibi et al., 2024) | LLM-extracted priors | LLM provides domain-knowledge constraints before statistical causal discovery runs | No structured knowledge base; single-source; no grounding module |
| CATE-B (Liu et al., 2024) | Literature queries (post hoc) | LLM queries scientific literature to resolve edge ambiguities after initial discovery | Literature queried only post hoc for disambiguation; no pre-discovery ontology |
| **This work** | **Structured ontology (Reactome) + LLM reasoning, with OmniPath cross-validation** | **Automatic variable grounding + REST-based ontology query (Reactome Content Service) + LLM causal reasoning from reaction roles → graded constraints, cross-validated against an aggregated multi-source baseline** | **Tri-source knowledge, grounded and auditable; isolates the contribution of structured mechanistic context vs. distilled multi-source consensus** |

### 1.3 Core novelty argument

Existing systems that use LLMs for causal discovery rely on implicit, parametric knowledge encoded in the LLM's weights at pretraining. This knowledge is unverifiable, domain-unspecific, and cannot be updated or audited. The proposed system instead uses a curated, peer-reviewed biological ontology (Reactome) as its primary knowledge source. The LLM acts as a reasoning bridge: it maps dataset variable names to ontology concepts (variable grounding) and then interprets structured reaction-participation data (inputs, outputs, catalysts, regulators) to infer directional causal claims with confidence scores.

This enables four claims that no existing system can make simultaneously:

1. Causal constraints are derived from an auditable, domain-authoritative source.
2. The system generalises to any proteomic dataset in human signalling without manual re-engineering — demonstrated on **two** datasets (Sachs primary, DREAM4 Predictive Signalling secondary).
3. The quality of ontology-derived constraints can be empirically measured against known ground-truth causal graphs, *and* cross-validated against an aggregated multi-source baseline (OmniPath) to attribute the gain to mechanistic context specifically.
4. The marginal value of statistical causal discovery over LLM judgments alone is empirically quantified: the same cached LLM claims are emitted as a predicted DAG directly (`C-LLM-only`) and compared against the LLM+CD pipeline. This addresses Kıcıman et al. (2023) — "Causal Reasoning and Large Language Models" — where LLMs alone are reported as strong pairwise causal-discovery baselines on Sachs.

### 1.4 Datasets

**Primary — Sachs et al. (2005).** 7,466 instances × 11 phosphorylated proteins/phospholipids in human immune cells, observational + interventional. Ground-truth DAG: 17 directed arcs. Canonical real-world causal-discovery benchmark; used as the running example in DoWhy, CDT, and dodiscover. Biology (MAPK, PI3K/Akt, calcium signalling) is well-covered by Reactome.

Loaded via CDT: `from cdt.data import load_dataset; data, graph = load_dataset('sachs')`.

**Caveat to report:** the bnlearn/CDT version of the Sachs DAG differs from the original Sachs et al. (2005) graph at ~3 edges. Metrics will be reported against both reference graphs.

**Secondary — DREAM4 Predictive Signalling Network Modelling.** ~7 antibody-measured human proteins (MCF7 breast cancer cell line, AKT/ERK1/MEK/p70S6K and adjacent signalling proteins) under stimulation/inhibition combinations. Real human data, Reactome-coverable. Used to test generalisation of the same pipeline to a second dataset in the same biology family without code changes.

**Tripwire (Day 1 of Step 2):** if Reactome coverage of either dataset's nodes drops below ~80%, fall back to OmniPath as the primary structured source for that dataset.

### 1.5 Knowledge sources

**Primary — Reactome.** Free, peer-reviewed, open-data knowledgebase of human pathways. Accessed via the public **Content Service REST API** (`https://reactome.org/ContentService`). Reactions are encoded with typed participation: input, output, catalystActivity (Reactome's `CatalystActivity` object — exposed inline by the REST `query/enhanced` endpoint with the catalyst's `physicalEntity` dereferenced), and regulatedBy (Reactome's `Regulation` / `PositiveRegulation` / `NegativeRegulation` objects — also returned inline with the regulator entity). The `/data/participants/{stId}` endpoint additionally flattens Complex / EntitySet membership into per-component UniProt and ChEBI references, eliminating the need for client-side recursion. Hierarchy: `Pathway → ReactionLikeEvent → PhysicalEntity → ReferenceEntity` (UniProt for proteins, ChEBI for small molecules).

**Secondary — OmniPath.** Aggregated meta-resource (Türei et al., *Nat. Methods* 2016, 2021) covering Reactome, SIGNOR, KEGG, and dozens of other sources with direction, sign, and per-source provenance. Python package `pypath-omnipath`. Used here to:
- Source the C0.5 floor ablation (structured-knowledge-only, no LLM).
- Provide a Reactome-only filtered view for a clean apples-to-apples no-LLM comparison.
- Act as a fallback if Reactome coverage of a given dataset is insufficient.

---

## 2. System Architecture

Sequential agentic pipeline, four logical stages. Each stage is a distinct Python module so components can be tested and ablated independently.

| # | Stage | Input → Output | Key technology |
|---|---|---|---|
| 1 | Variable Grounding | Column names + dataset description → `{column: {uniprot_id \| chebi_id \| list[uniprot_id], display_name, confidence}}` | LLM structured output (JSON mode) + Reactome/OmniPath validation |
| 2 | Ontology Query | Grounded variables → reaction subgraphs with typed roles (input, output, catalyst, regulator) | Reactome Content Service REST API via `httpx`, with on-disk response cache; OmniPath Python API for cross-source queries |
| 3 | Causal Reasoning | Reaction context → `(cause, effect, confidence, constraint_type, reasoning_trace)` per variable pair | LLM with chain-of-thought; structured JSON output; **cached** by `(model_id, prompt_hash)` |
| 4 | Constrained Discovery | Graded constraints + observational data → discovered DAG + evaluation metrics | `causal-learn` (PC, GES, LiNGAM) + `BackgroundKnowledge` API (PC), prior matrix (LiNGAM), post-hoc edits with explicit caveat (GES) |

Cross-cutting infrastructure (shared across stages):

- **On-disk response cache** (`cache/llm/`, `cache/reactome/`) — single JSON-per-key, ~30 lines of code total. The same `call_with_cache(...)` pattern wraps both the OpenRouter LLM client and the Reactome REST client. Default replay-from-cache; hits the network only on miss. Both caches committed to the repo so the pipeline can be replayed without an API key.
- **Experiment runner** (`experiments/runner.py`) — orchestrates conditions × algorithms × seeds.
- **Results store** (`experiments/results/*.json`) — typed artefacts per condition × algorithm × seed.

---

## 3. Development Steps

Each step is self-contained, produces a testable artefact, maps to one or more modules. Steps are ordered by dependency.

### Step 1 — Sachs Baseline + LLM Cache Layer (Foundation)

**Goal.** Reproduce PC / GES / LiNGAM baseline on Sachs; build the LLM cache layer that will be used by every subsequent step.

**Output.**
- `experiments/baseline_sachs.json` with mean ± std of SHD, AUPR, Precision, Recall, F1 across 10 seeds for PC / GES / LiNGAM.
- `llm/cache.py` with full-coverage unit tests.

**Depends on.** Nothing.

**Effort.** 1.5–2 days.

**Implementation tasks.**
- Install dependencies via `task install`: ensure `causal-learn`, `cdt`, `httpx`, `pypath-omnipath`, `openai` are pinned in `pyproject.toml`.
- Load Sachs via CDT.
- Implement evaluation harness in `evaluation/harness.py`: takes two `networkx.DiGraph` objects (predicted, true), returns dict of metric values. Reuse `evaluation/metrics.py`.
- Run PC (α=0.05, fisherz), GES (BIC), DirectLiNGAM with default hyperparameters, 10 seeds (0–9).
- Build `llm/cache.py` (lightweight, ~30 lines, single JSON-per-key under `cache/llm/`):
  - `LLMCache(cache_dir)` — JSON-on-disk store, `get(model_id, messages) -> response | None`, `put(model_id, messages, response)`.
  - Cache key = `sha256(json.dumps({"model": model_id, "messages": messages}, sort_keys=True))`.
  - `LLMClient` wrapper around the OpenAI-compatible client that consults cache before calling.
- Keep the existing model setting (`openrouter/free`). Record the per-request `served_model` from each OpenRouter response into the cache so the paper appendix can report which underlying providers served the requests.

**Verification.**
- PC SHD on Sachs should land in the published range (~17–22 without priors, depending on metric variant).
- Cache layer test: run any prompt twice, second run must not call the API.

---

### Step 2 — Reactome REST Client (Foundation)

**Goal.** Build a thin Python client over the Reactome Content Service REST API with on-disk caching, sufficient for protein-protein and protein-metabolite co-participation queries with role information.

**Output.**
- `reactome/client.py` with `ReactomeClient` class.
- `reactome/queries.cypher` with the validated query set, commented.
- `experiments/reactome_coverage.json` — coverage report for Sachs and DREAM4 nodes.

**Depends on.** Step 1 (need dataset variable names to validate coverage).

**Effort.** 3–4 days.

**Schema notes (corrected from v1).**
- `(rle:ReactionLikeEvent)-[:input]->(pe:PhysicalEntity)` — direct.
- `(rle)-[:output]->(pe)` — direct.
- `(rle)-[:catalystActivity]->(ca:CatalystActivity)-[:physicalEntity]->(pe)` — **2 hops** via `CatalystActivity`.
- `(reg:Regulation|PositiveRegulation|NegativeRegulation)-[:regulator]->(pe)` and `(reg)-[:regulatedEntity]->(rle)` — Regulation is a separate node, not a relationship.
- Species filter: `(rle)-[:species]->(:Species {taxId:'9606'})` — `Species` is a node connected by `[:species]`, not a relationship property.
- `PhysicalEntity` types: `EntityWithAccessionedSequence` (proteins, has `[:referenceEntity]->(:ReferenceGeneProduct)` carrying UniProt `identifier`), `SimpleEntity` (small molecules, has `[:referenceEntity]->(:ReferenceMolecule)` carrying ChEBI `identifier`), `Complex`, `EntitySet`.

**Implementation tasks.**
- Confirm REST connectivity: `GET https://reactome.org/ContentService/data/discover/R-HSA-5672978` returns 200 OK.
- Implement `reactome/client.py` over `httpx`, reusing the `cached_call` helper from Step 1. Two endpoints carry the bulk: `/data/query/enhanced/{stId}` (full reaction with role-typed participants inline) and `/data/participants/{stId}` (Complex membership flattened to UniProt / ChEBI accessions).
- See `docs/step_02_reactome_setup.md` for the full endpoint reference and verified examples.
- Implement `ReactomeClient`:
  - `__init__(uri, user, password)`, `close()`.
  - `get_reaction_context(entity_a: EntityRef, entity_b: EntityRef) -> list[ReactionRecord]` where `EntityRef` carries `kind: "protein" | "metabolite"` and `id: str | list[str]` (lists support families like PKC).
  - For each co-participating reaction, return `{reaction_name, reaction_id, pathway, role_a, role_b, reaction_type}`.
  - `get_entity_info(ref: EntityRef) -> dict` — display name, gene names, pathways.
- Implement `format_context_for_llm(records, name_a, name_b) -> str` — natural-language paragraph for prompt inclusion.
- Run coverage check: for Sachs's 11 nodes and DREAM4 PSN's 7 nodes, report fraction with at least one Reactome physical entity, mean reactions per node, mean co-participations per pair.

**Tripwire.** If Reactome coverage < ~80% on either dataset, escalate to OmniPath-as-primary for that dataset. Capture the decision in `experiments/reactome_coverage.json`. Note: this tripwire is keyed to *node* coverage, not pair coverage. Low reaction-level pair coverage driven by curation modality (e.g., PKC/PKA regulation modelled at pathway level) is addressed by Step 2.5 below, not by escalating to OmniPath.

---

### Step 2.5 — Reactome Context Expansion (Foundation, post-coverage)

**Goal.** Close the documented Sachs pair-coverage gap (~20% reaction-level → target ~50–60% pathway-level) without altering Reactome as the primary source. Driven by the empirical finding in `experiments/reactome_coverage.json` that PKC- and PKA-mediated regulation is curated at pathway scope, not reaction scope, so the reaction-level intersection is empty for many true edges (e.g. `(PKC, praf)`, `(PKA, pmek)`).

**Output.**
- `reactome/client.py` extended with three additive methods (no breaking change to existing signatures): `pathways_for_entity`, `regulator_chain_records`, `co_complex_records`.
- `experiments/reactome_coverage.json` re-emitted with a `pair_coverage_by_layer` breakdown (`reaction`, `co_pathway`, `regulator_chain`, `co_complex`, `union`) so the LLM in Step 4 has an audit trail for which evidence layer fired for each pair.
- `format_context_for_llm` updated to phrase each evidence layer distinctly so the Step 4 LLM can weight them.

**Depends on.** Step 2.

**Effort.** ~1 day.

**Implementation tasks.**
- `pathways_for_entity(EntityRef) -> set[str]` — harvest the `eventOf` already returned inline by `query/enhanced` for each entity's reactions; cache via the existing `cached_call` helper. Use it to emit `co_pathway` records when `a` and `b` share at least one Reactome `Pathway`.
- `regulator_chain_records(a, b) -> list[ReactionRecord]` — for every reaction producing `b`, walk `regulatedBy → regulator` and `catalystActivity → physicalEntity`, expand via `participants`, and emit a record with `role_a="regulator"` (or `"catalyst"`) and `role_b="output"` if any expanded refEntity hits `a.normalised_ids`. Reuses existing `_resolve_regulation` and `_matches_for_pe`. Sign carries through.
- `co_complex_records(a, b) -> list[ReactionRecord]` — index Complex `refEntities` from cached `participants` payloads; emit a `role_a=role_b="co_complex"` record where both accessions co-occur in the same PE without a shared reaction. New `reaction_type="ComplexMembership"`.
- Extend the `Role` literal with `"co_pathway"` and `"co_complex"`; update `format_context_for_llm` per-layer phrasing ("co-occur in the SCF complex", "PKA positively regulates a reaction producing MEK1", etc.).

**Acceptance criteria.**
- Sachs `pair_coverage_by_layer.union ≥ 0.55` (target; not a hard tripwire).
- `(PKC, praf)`, `(PKA, pmek)`, and `(PIP3, plcg)` each return ≥ 1 record from at least one layer.
- All existing `tests/test_reactome_client.py` tests pass unchanged; new tests cover at minimum: a positive- and a negative-regulator path producing signed `ReactionRecord`s, the `/search/query` metabolite path, the retry-with-backoff on transient 5xx, and one `co_pathway` and one `co_complex` record.

**Out of scope.** Any LLM call. OmniPath. Sign-aware regulator weighting (Step 4 owns that).

---

### Step 3 — Variable Grounding (Foundation)

**Goal.** Map dataset column names to Reactome-resolvable identifiers via LLM, validated against Reactome.

**Output.**
- `grounding/ground.py` with the grounding function.
- `experiments/grounding_sachs.json` and `experiments/grounding_dream4.json`.

**Depends on.** Step 2.5 (Reactome validation uses the expanded context layers).

**Effort.** 2–3 days.

**Generalised output schema (per column).**

```json
{
  "<column_name>": {
    "kind": "protein" | "metabolite" | "family",
    "ids": ["P15056"] | ["CHEBI:18348"] | ["P17252", "Q05655", "Q02156"],
    "canonical_name": "RAF1" | "PIP2" | "Protein Kinase C (pan-isoform)",
    "gene_names": ["RAF1"],
    "confidence": 0.92,
    "reasoning": "...",
    "reactome_validated": true
  }
}
```

**Implementation tasks.**
- Extend the prompt to handle phospho-prefix conventions (`p` = phosphorylated state), protein families (return list of canonical isoforms), and metabolites (return ChEBI IDs).
- Provide few-shot examples covering each kind: a single protein (`praf` → RAF1, P15056), a family (`pkc` → PRKCA/B/G/D/E), a metabolite (`PIP2` → CHEBI:18348).
- Few-shot the prompt to return BOTH protonated and deprotonated ChEBI forms for phosphoinositides (PIP2: `CHEBI:18348` + `CHEBI:58456`; PIP3: `CHEBI:16618` + `CHEBI:57836`). Single-form returns systematically halve metabolite reaction hits in Reactome.
- Validate every returned ID against Reactome via `ReactomeClient.get_entity_info`. If the entity isn't in Reactome, mark `reactome_validated: false` and set `confidence` to `min(reported, 0.4)`.
- Cache the grounding LLM call via `llm/cache.py`.
- Hand-curate gold-standard groundings for Sachs and DREAM4 PSN, evaluate accuracy against gold standard, log failures.

**Sachs gold-standard grounding policy.** Lift the existing `experiments/reactome_coverage.py:SACHS_GROUNDING` dict as the Sachs gold-standard fixture, with two confirmed adjustments before scoring:
1. Drop PKA regulatory subunits (`P10644`, `P31321`) — Sachs measures activated PKA = catalytic subunit only.
2. Drop AKT3 (`Q9Y243`) from `pakts473` — not meaningfully expressed in the Sachs immune-cell context.

These adjustments are pinned now to avoid a "regression" appearance when Step 3's LLM returns the smaller, biologically-correct set.

---

### Step 4 — Causal Reasoning from Reaction Context (Core)

**Goal.** LLM reasons over Reactome reaction context to produce directional causal claims with graded confidence.

**Output.**
- `reasoning/reason.py`.
- `experiments/causal_priors_sachs.json`, `experiments/causal_priors_dream4.json`.
- Per-pair reasoning trace stored alongside for auditability and ablation.

**Depends on.** Steps 2 and 3.

**Effort.** 4–6 days.

**Reasoning rules baked into the prompt.**
- `input → output` within the same reaction is the strongest causal signal.
- `catalyst` for a reaction where the other entity is `output` implies activation / causal influence.
- `regulator` of a reaction producing the other entity implies causal influence (sign carried separately).
- Both entities as `input` in the same reaction: direction is *not* established — return `unknown`.
- Multiple consistent reactions across pathways increase confidence; contradictory roles decrease it.

**Confidence scale.** 0.9–1.0 = well-established mechanism; 0.7–0.89 = likely based on pathway context; 0.5–0.69 = plausible but uncertain; < 0.5 = `unknown` (discarded).

**Constraint type decision rule.**
- `confidence ≥ 0.9` → `hard_required` (or `hard_forbidden_reverse` if direction is *forbidden*).
- `0.6 ≤ confidence < 0.9` → `soft_prior`.
- `confidence < 0.6` → `unknown` (discarded).

**Implementation tasks.**
- For each ordered pair `(i, j)` of grounded variables, call `ReactomeClient.get_reaction_context` → `format_context_for_llm`.
- Call the causal-reasoning prompt; parse structured JSON.
- Cache every call.
- Aggregate per ordered pair (when the LLM sees the pair in both orders, prefer the higher-confidence directional call; if they conflict, take confidence-weighted majority).
- Save full output including reasoning traces.
- For the **C0.5 floor ablation**, in parallel produce a no-LLM constraint set:
  - For each pair, query OmniPath for direct A→B edges (filtered to `is_directed=True`, optionally to `source=Reactome` for the apples-to-apples view).
  - Emit a constraint with confidence proportional to source consensus (number of independent OmniPath sources supporting the edge, normalised).
  - This is a ~30-line addition that falls out of the OmniPath integration and provides the LLM-free floor without additional LLM cost.

**Out of scope (parking lot).** Deterministic rule-based reasoner over the Reactome subgraphs. Possible stretch goal if the C0.5 OmniPath floor turns out to be too coarse to be informative.

---

### Step 5 — Constraint Integration and Constrained Discovery (Core)

**Goal.** Translate causal priors into `causal-learn` `BackgroundKnowledge` (PC), prior matrices (LiNGAM), and post-hoc edits (GES, with explicit caveat).

**Output.**
- `constraints/builder.py` (extends current `constraints/constraint_builder.py`).
- `experiments/discovery_results.json` with metrics per `(condition × algorithm × seed)`.

**Depends on.** Steps 1, 3, 4.

**Effort.** 2 days.

**Implementation tasks.**
- Implement `ConstraintBuilder` accepting `causal_priors.json` + column names + `confidence_threshold`. Constraint thresholds will be swept at 0.6, 0.7, 0.8, 0.9 plus oracle.
- For each algorithm:
  - **PC.** Use `BackgroundKnowledge.add_required_by_node` / `add_forbidden_by_node`. Native pathway, supported in `causal-learn`.
  - **LiNGAM.** Build the `(n × n)` prior matrix (`1` required, `0` forbidden, `-1` unknown — note: this differs from v1 which used `-1` as default; the codebase's current convention is correct, retain it).
  - **GES.** No native prior-knowledge hook in current `causal-learn`. **Apply constraints post-hoc as direct edge edits, and report this as a methodological caveat in the paper.** Optionally explore `pytetrad` fGES integration as a stretch goal; do not block on it.
- Implement graph serialisation: adjacency matrices + edge lists + GraphML for reproducibility.

---

### Step 6 — Ablation Study and Evaluation (Evaluation)

**Goal.** Quantify the contribution of each pipeline component on both datasets; produce publication-ready results.

**Output.**
- `experiments/ablation_results.json`.
- `experiments/constraint_quality.json`.
- `figures/` — paper-ready figures.
- `tables/` — LaTeX-formatted tables.

**Depends on.** Steps 1–5.

**Effort.** 3–4 days.

**Ablation conditions (revised).**

| ID | Condition | What it measures |
|----|-----------|------------------|
| C0 | No knowledge | Floor — algorithms only |
| **C0.5** | **OmniPath direct edges only, no LLM** | **Isolates the LLM contribution from structured-knowledge contribution** |
| **C1** | **Free-text-LLM (current skeleton on a Sachs / DREAM4 background paragraph)** | **Isolates Reactome contribution from plain LLM-on-text** |
| C2 | Reactome+LLM, hard constraints (τ ≥ 0.9) | High-precision setting |
| C3 | Reactome+LLM, hard + soft (τ ≥ 0.7) | Default proposed setting |
| C4 | Reactome+LLM, hard + soft (τ ≥ 0.6) | Threshold-sensitivity test |
| **C-LLM-only** | **Reactome+LLM judgments emitted as DAG directly, no CD algorithm** | **Isolates whether causal discovery adds value beyond LLM judgments** |
| C5 | Oracle (true edges as required constraints) | Ceiling — maximum possible improvement |

Conditions C0 through C5 are run 10 times per algorithm (PC / GES / LiNGAM) with seeds 0–9, on both datasets. C-LLM-only produces a single deterministic DAG per dataset (it has no algorithm dimension and no seed dimension), reusing the same cached LLM priors that drive C2–C4. Mean ± std reported for the algorithmic conditions; point estimate for C-LLM-only.

**Constraint Quality Evaluation (independent of discovery).**

A novel secondary contribution: how accurately can an LLM infer causal direction from ontological reaction context, evaluated against ground truth in isolation from the downstream causal-discovery algorithm.

- **Precision** of extracted causal edges (fraction of predicted edges that match ground truth).
- **Recall** of extracted causal edges (fraction of ground-truth edges covered).
- **Hallucination rate** (fraction of extracted edges that contradict ground truth — wrong-direction calls).
- **Coverage rate** (fraction of variable pairs for which Reactome returned non-empty reaction context).
- **Cost / latency report** (tokens, dollars, wall-clock per pipeline stage).

**Headline number to report.** *Fraction of the C0 → C5 oracle gap closed by the proposed Reactome+LLM pipeline*, broken down per algorithm and per dataset, with comparison against C0.5, C1, and C-LLM-only to attribute the gain to (a) the structured ontology, (b) the LLM's reasoning over it, and (c) the causal-discovery step on top.

---

### Step 7 — DREAM4 Predictive Signalling Integration (Evaluation)

**Goal.** Demonstrate that the same pipeline, with no code changes, generalises to a second human signalling dataset.

**Output.**
- `experiments/grounding_dream4.json`, `experiments/causal_priors_dream4.json`, `experiments/discovery_results_dream4.json`.

**Depends on.** Steps 1–6 complete on Sachs.

**Effort.** 2–3 days.

**Implementation tasks.**
- Acquire DREAM4 Predictive Signalling data and ground-truth network. Verify availability on Day 1.
- Hand-curate the DREAM4 PSN gold-standard grounding (~7 nodes).
- Run Steps 3–6 on DREAM4 PSN with no code changes — only configuration changes (dataset path, gold-standard grounding file).
- Report cross-dataset performance.

**Tripwire.** If DREAM4 PSN data is not accessible (paywalled, unmaintained download), substitute synthetic SCM data sampled from a Reactome-derived human pathway DAG (PI3K/Akt or RAS/MAPK subnetwork). Document the substitution clearly.

---

### Step 8 — Analysis, Paper Writing, Reproducibility (Evaluation)

**Goal.** Interpret results scientifically; produce paper draft and a fully reproducible code repository.

**Output.**
- Paper draft — Introduction, Method, Experiments, Results, Discussion.
- Public GitHub repository with pinned dependencies, committed LLM and Reactome caches, regenerable figures. No Docker required.

**Depends on.** Steps 6 and 7.

**Effort.** 2–3 weeks.

**Key research questions to address in the paper.**
- How much does ontology-derived domain knowledge improve causal discovery accuracy over no-knowledge baselines, and which algorithm benefits most?
- What fraction of ground-truth edges is recoverable from Reactome reaction context alone, and what is the hallucination rate?
- **Does LLM reasoning over Reactome subgraphs add value beyond structured-knowledge consensus alone (C2/C3 vs C0.5)?**
- **Does Reactome-grounded reasoning add value beyond plain free-text-LLM domain knowledge (C2/C3 vs C1)?**
- **Does causal discovery add value beyond LLM judgments alone (C2/C3 vs C-LLM-only)? If LLM-only is competitive, the paper's framing shifts toward "LLM-as-causal-reasoner with auditable evidence" rather than "LLM-augmented CD."**
- How sensitive is performance to the confidence threshold τ?
- How do results compare to CausalCopilot's LLM-pruned output (where available from their paper) and Kıcıman et al. (2023)'s LLM-only causal-direction results on Sachs?

**Reproducibility checklist.**
- `pyproject.toml` with pinned dependency versions.
- Fixed random seeds in all stochastic components.
- LLM model: pinned to `Qwen/Qwen3-235B-A22B` on a Baseten dedicated deployment (OpenAI-compatible API). `BASETEN_API_KEY` is the credential; the deployment-specific base URL lives in `llm/client.py:LLM_BASE_URL` (overridable via the `LLM_BASE_URL` env var if the deployment is recreated). The earlier OpenRouter `openrouter/free` routing alias and Nemotron pin are obsolete (see changelog item 5 for the model-selection trail). Per-request `served_model` is still recorded in the cache for the paper appendix.
- Full LLM and Reactome response caches committed to repo, kept in lockstep with code (committed in the same PR as the code that produced them; not deferred to Step 8). The pipeline replays end-to-end from cache without any API key.
- Full `grounding_*.json` and `causal_priors_*.json` committed.
- All ablation results and figures regenerable from a single command (`task ablation`).
- Cost / latency report in the appendix (where applicable for free-tier).

---

## 4. Step Dependencies and Timeline

| Step | Name | Realistic effort | Blocking |
|------|------|------------------|----------|
| 1 | Sachs baseline + LLM cache layer | 1.5–2 days | None |
| 2 | Reactome REST client + coverage | 2–3 days | Step 1 (variable names) |
| 2.5 | Reactome context expansion (pathway / regulator-chain / co-complex) | ~1 day | Step 2 |
| 3 | Variable grounding (proteins + metabolites + families) | 2–3 days | Step 2.5 |
| 4 | Causal reasoning + caching + OmniPath floor | 4–6 days | Steps 2 + 3 |
| 5 | Constraint integration + constrained discovery | 2 days | Steps 1 + 3 + 4 |
| 6 | Ablation study + constraint quality eval | 3–4 days | Steps 1–5 |
| 7 | DREAM4 Predictive Signalling integration | 2–3 days | Steps 1–6 |
| 8 | Paper writing + reproducibility wrap | 2–3 weeks | Steps 6 + 7 |

**Total: 6–8 weeks of focused implementation + writing.**

Steps 1 and 2 cannot be parallelised here because Step 2's coverage check requires Step 1's variable list. The critical path is Step 1 → Step 2 → Step 2.5 → Step 3 → Step 4 → Step 5 → Step 6 → Step 7 → Step 8.

---

## 5. Venue Strategy

| Venue | Deadline (typical) | Fit | Recommended? |
|-------|--------------------|-----|--------------|
| RAI 2026 | June–July 2026 | Continuation of two prior RAI 2025 papers; lowest risk, lowest ceiling | Feasible only with aggressive schedule; Steps 1–6 done by mid-June |
| **CLeaR 2027** | **October 2026** | **Audience match is excellent; LLM-augmented causal discovery papers fit cleanly** | **Primary recommendation** — comfortable runway, right audience |
| ECAI 2027 | April 2027 | Systems papers welcome at ECAI; original v1 venue choice, one cycle later | Strong fallback |
| AAAI 2027 | August 2026 | Ambitious for systems-style work; strong ML methods bias | Stretch |
| Bioinformatics / NPJ Systems Biology | Rolling | Foreground the biology; longer paper format | Parallel option (preprint first, journal in background) |
| ECAI 2026 | Late April 2026 (passed) | — | **Missed** |
| UAI / AISTATS | Various | Methods novelty bar is too high for a systems+benchmark paper | Not recommended |

**Default plan: target CLeaR 2027 with arXiv preprint mid-2026.**

---

## 6. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| 1 | Reactome REST API rate-limited, unstable, or schema drift | Low | High | On-disk cache makes reruns offline; retry-with-backoff in the client; Homebrew-Neo4j fallback documented if REST insufficient (Step 2) |
| 2 | Reactome *node* coverage of dataset < 80% | Low (Sachs — confirmed 11/11), Medium (DREAM4 — unverified) | High | Tripwire in Step 2; OmniPath as primary if triggered. Keyed to node coverage, not pair coverage. |
| 3 | DREAM4 Predictive Signalling data inaccessible | Medium | Medium | Synthetic-from-Reactome-pathway substitution |
| 4 | LLM costs balloon | Low | Low | Caching is mandatory; ~110 ordered pairs per dataset is small |
| 5 | LLM nondeterminism breaks reproducibility | Medium without caching, Low with | High | Cache on first run, replay-from-disk thereafter; record `served_model` per request; cache committed |
| 6 | Post-hoc required/forbidden injection confounds the comparison across all three CD algorithms | High | Medium | **PC, GES, LiNGAM** all apply constraints post-hoc with `apply_post_hoc_edits`. **GES**: no native prior hook in `causal-learn`. **PC (May 2026)**: Fisher-Z can drop a required edge before orientation, so required edges are injected after PC runs; cycle-closing drops are logged in `dropped_due_to_cycle` and `pc_post_hoc_*`. **LiNGAM (May 2026, PR3b)**: the native `prior_knowledge` matrix with `apply_prior_knowledge_softly=True` leaves offending coefficients above the 0.001 threshold, so it does not actually enforce forbidden edges; switched to the same post-hoc path as GES + PC (`lingam_prior_mode="post_hoc"`), with cycle drops logged in `dropped_due_to_cycle`. Report the uniform post-hoc treatment as the headline methodological caveat; optionally explore `pytetrad` fGES + native-LiNGAM modes as future work. |
| 7 | Sachs ground-truth contestation | Medium | Low | Report against both bnlearn/CDT and original 2005 graphs |
| 8 | OmniPath floor (C0.5) is too coarse to inform | Low | Low | Stretch fallback: deterministic rule-based reasoner over Reactome subgraphs |
| 9 | Single-dataset reviewer pushback (despite DREAM4 secondary) | Medium | Medium | Cross-source attribution via OmniPath ablation strengthens generalization claim |
| 10 | Timeline slips past CLeaR 2027 deadline | Medium | Low | ECAI 2027 fallback adds 6 months |
| 11 | C-LLM-only outperforms LLM+CD pipeline | Medium | Low (acceptable outcome) | Reframe paper as "LLM-as-causal-reasoner with auditable Reactome evidence"; the CD comparison becomes a quantified negative result that is itself a contribution. No code changes required. |
| 12 | Reactome reaction-level pair coverage low because of curation modality (PKC/PKA modelled at pathway level, not per-reaction) | Confirmed on Sachs (`pair_coverage = 11/55`, ground-truth-edge coverage = 5/18) | Medium | Step 2.5 pathway / regulator-chain / co-complex expansion; OmniPath C0.5 floor in Step 4 as second check; document in paper limitations. Does NOT trigger OmniPath-as-primary (Risk #2 is keyed to node coverage, not pair coverage). |

---

## 7. Open Items / Parking Lot

1. **Streamlit UI fate.** `main.py` currently runs the LUCAS skeleton. Decide after Step 6 whether to (a) retire it, (b) repurpose it as a results explorer over `ablation_results.json`, or (c) keep the LUCAS skeleton as the C1 free-text-LLM baseline. Default: keep until the C1 ablation is implemented, then re-evaluate.
2. **Deterministic rule-based Step-4 reasoner.** Out of scope for v2; possible stretch goal if the OmniPath C0.5 floor is too coarse to attribute the LLM's contribution.
3. **GES with native priors via pytetrad fGES.** Stretch goal; document as future work if not pursued.
4. **Cross-ontology comparison (Reactome-only OmniPath view vs full multi-source OmniPath view).** Falls out almost for free from the Step-4 implementation; report if results are interesting.

---

## 8. Selected Agent Prompts

### 8.1 Variable Grounding (revised)

```text
SYSTEM
You are a bioinformatics expert in proteomics and cell signalling. Your task is
to map dataset column names to canonical biomedical identifiers.

Rules:
- Prefix 'p' typically denotes a phosphorylated measurement (e.g. praf = phospho-RAF1).
- Return human (Homo sapiens) UniProt IDs for proteins, ChEBI IDs for metabolites/lipids.
- For protein families measured by pan-isoform antibodies (e.g. PKC, PKA, ERK1/2),
  return a LIST of canonical UniProt IDs covering the major isoforms.
- For phosphoinositide metabolites (PIP2, PIP3), return ChEBI IDs.
- Report a confidence (0.0-1.0) per column.
- If you cannot map with confidence > 0.5, return null for `ids`.

Output ONLY a JSON object. No preamble, no markdown.

USER
Dataset: {dataset_description}
Domain: {domain_hint}
Columns: {column_list}

Return a JSON object with this exact schema:
{
  "<column_name>": {
    "kind": "protein" | "metabolite" | "family",
    "ids": ["<UniProt or ChEBI accession>", ...] | null,
    "canonical_name": "<human-readable name>",
    "gene_names": ["<HGNC gene symbol>", ...],
    "confidence": <float 0.0-1.0>,
    "reasoning": "<one sentence explaining the mapping>"
  }
}
```

### 8.2 Causal Reasoning (unchanged in spirit, refined for families/metabolites)

```text
SYSTEM
You are a causal-reasoning expert in molecular cell biology. You are given
structured information about how two biomolecules co-participate in biochemical
reactions, retrieved from the Reactome pathway database. Your task is to reason
about the causal direction between them and assign a confidence score.

REASONING RULES
- 'input → output' within the same reaction is the strongest causal signal
  (upstream → downstream).
- 'catalyst' for a reaction where the other entity is 'output' implies activation
  / causal influence.
- 'regulator' of a reaction producing the other entity implies causal influence.
- Both entities as 'input' in the same reaction does NOT establish direction —
  return 'unknown'.
- Multiple consistent reactions across pathways increase confidence; contradictory
  roles decrease confidence.
- For protein families, consider co-participation of any isoform.

CONFIDENCE SCALE
- 0.9-1.0 — well-established mechanism (multiple consistent reactions).
- 0.7-0.89 — likely based on pathway context.
- 0.5-0.69 — plausible but uncertain.
- < 0.5  — return direction as 'unknown'.

Output ONLY valid JSON. No preamble.

USER
Entity A: {name_a} ({ids_a})
Entity B: {name_b} ({ids_b})

Reactome reaction context:
{formatted_context}

Determine the most likely causal direction. Return:
{
  "cause": "<name_a or name_b or 'unknown'>",
  "effect": "<name_a or name_b or 'unknown'>",
  "confidence": <float 0.0-1.0>,
  "constraint_type": "hard_required" | "soft_prior" | "hard_forbidden_reverse" | "unknown",
  "supporting_reactions": ["<reaction name>", ...],
  "contradicting_reactions": ["<reaction name>", ...],
  "reasoning": "<2-3 sentences>"
}
```

---

## 9. Repo Layout (target)

```
text2causal/
├── docs/
│   ├── dev_plan_v2.md           # this file
│   └── reactome_schema_notes.md # produced in Step 2
├── llm/
│   ├── cache.py                 # NEW: response cache layer
│   ├── client.py                # NEW: cached LLM client wrapper
│   ├── prompts.py               # extended with new prompts
│   └── extract_relations.py     # current free-text extractor (kept for C1)
├── reactome/
│   ├── client.py                # NEW
│   └── queries.cypher           # NEW
├── omnipath/
│   └── client.py                # NEW
├── grounding/
│   └── ground.py                # NEW
├── reasoning/
│   └── reason.py                # NEW
├── constraints/
│   └── builder.py               # extended from constraint_builder.py
├── causal_discovery/            # unchanged in shape
├── evaluation/
│   ├── metrics.py               # current
│   └── harness.py               # NEW
├── experiments/
│   ├── runner.py                # NEW
│   └── results/                 # produced artefacts
├── data/
│   ├── lucas/                   # current
│   ├── sachs/                   # NEW
│   └── dream4_psn/              # NEW
├── tests/                       # unittest
├── cache/                       # NEW: response caches (LLM + Reactome), committed
├── pyproject.toml
└── Taskfile.yml
```

---

## 10. Key References

- Sepioło, D. & Ligęza, A. *Towards Model-Driven Explainable Artificial Intelligence: Function Identification with Grammatical Evolution.* Applied Sciences, 14(13):5950, 2024.
- Jarosławski, M., Sepioło, D. & Ligęza, A. *From Data to Decisions: Comparing Causal Discovery Methods on a Benchmark Dataset.* RAI 2025.
- Sepioło, D., Jarosławski, M. & Ligęza, A. *Grammar Refinement in Grammatical Evolution Using Large Language Models.* RAI 2025.
- Wang et al. *CausalCopilot: An Agentic System for Automated Causal Analysis.* arXiv:2504.13263, 2025.
- Khatibi et al. *ALCM: Autonomous LLM-augmented Causal Discovery.* 2024.
- Liu et al. *CATE-B.* 2024.
- Milacic et al. *The Reactome Pathway Knowledgebase 2024.* Nucleic Acids Research, 2023.
- Türei et al. *OmniPath: guidelines and gateway for literature-curated signalling pathway resources.* Nature Methods, 2016. *Integrated intra- and intercellular signaling knowledge for multicellular omics analysis.* Mol. Syst. Biol., 2021.
- Sachs et al. *Causal Protein-Signaling Networks Derived from Multiparameter Single-Cell Data.* Science, 308(5721):523–529, 2005.
- Zheng et al. *Causal-learn: Causal Discovery in Python.* JMLR, 25(60):1–8, 2024.
- Shimizu et al. *A Linear Non-Gaussian Acyclic Model for Causal Discovery.* JMLR, 7:2003–2030, 2006.
- Prill et al. *Towards a Rigorous Assessment of Systems Biology Models: The DREAM3 Challenges.* PLoS ONE, 5(2):e9202, 2010 (DREAM4 PSN reference).
