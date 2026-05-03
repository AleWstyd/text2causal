# Step 3 — Variable Grounding

**Role.** Foundation.

**Goal.** Map dataset column names to Reactome-resolvable identifiers (UniProt for proteins, ChEBI for metabolites, lists for protein families), validated against Reactome.

**Depends on.** Step 1 (cached LLM client), Step 2 (`ReactomeClient` for validation).

**Effort.** 2–3 days.

---

## Deliverables

| Artefact | Purpose |
|----------|---------|
| `grounding/ground.py` | `ground_columns(...)` |
| `grounding/gold_sachs.json` | Hand-curated gold-standard grounding for Sachs (committed) |
| `experiments/grounding_sachs.json` | LLM-produced + Reactome-validated grounding (committed) |
| Tests in `tests/test_grounding.py` | Validation logic, accuracy vs gold |

---

## Output schema (per column)

```json
{
  "<column_name>": {
    "kind": "protein" | "metabolite" | "family",
    "ids": ["P15056"] | ["CHEBI:18348"] | ["P17252", "Q05655", "Q02156"] | null,
    "canonical_name": "RAF1",
    "gene_names": ["RAF1"],
    "confidence": 0.92,
    "reasoning": "...",
    "reactome_validated": true
  }
}
```

`null` ids = ungrounded (LLM declined to map with confidence > 0.5, or all returned ids failed Reactome validation).

---

## Implementation tasks

1. **Hand-curate Sachs gold standard** (`grounding/gold_sachs.json`). 11 entries. Use canonical UniProt/ChEBI IDs from the literature. Document the call for each family member.

   Reference values (verify against UniProt / ChEBI before committing):

   | Sachs column | kind | ids | canonical |
   |--------------|------|-----|-----------|
   | `praf` | protein | `["P04049"]` | RAF1 |
   | `pmek` | protein | `["Q02750"]` | MAP2K1 |
   | `plcg` | protein | `["P19174"]` | PLCG1 |
   | `PIP2` | metabolite | `["CHEBI:18348"]` | PtdIns(4,5)P2 |
   | `PIP3` | metabolite | `["CHEBI:16618"]` | PtdIns(3,4,5)P3 |
   | `p44/42` | family | `["P27361", "P28482"]` | ERK1/2 (MAPK3, MAPK1) |
   | `pakts473` | protein | `["P31749"]` | Akt1 |
   | `PKA` | family | `["P17612", "P22694"]` | PKA catalytic α/β |
   | `PKC` | family | `["P17252", "P05771", "Q05655", "Q02156", "Q05513"]` | PKCα/β/δ/ε/ζ |
   | `P38` | protein | `["Q16539"]` | MAPK14 |
   | `pjnk` | family | `["P45983", "P45984"]` | JNK1/2 (MAPK8, MAPK9) |

   These are reference values — verify each before committing. The exact column names should be matched against `cdt.data.load_dataset('sachs')` — they may be slightly different.

2. **Implement the grounding prompt.** Use the prompt from `docs/dev_plan_v2.md` §8.1. Few-shot examples should cover:
   - Single protein with phospho prefix (`praf` → RAF1).
   - Family (`PKC` → list of 5 isoforms).
   - Metabolite (`PIP2` → ChEBI).
   - An ambiguous case where the right answer is `null`.

3. **Implement `grounding/ground.py`.**
   ```python
   def ground_columns(
       columns: list[str],
       dataset_description: str,
       domain_hint: str,
       llm_client: LLMClient,
       reactome_client: ReactomeClient,
   ) -> dict[str, GroundingRecord]: ...
   ```
   - Build the prompt with the column list, dataset description, and domain hint.
   - Call the cached LLM client.
   - Parse the JSON response.
   - **Gene→UniProt resolution (added after Step 3 implementation surfaced the open-weight UniProt-recall failure mode).** For every protein/family column where the LLM returned `gene_names`, look each gene name up in Reactome's `/search/query?types=Protein&species=Homo+sapiens` index, take the canonical Homo sapiens UniProt for each gene (filtered by `databaseName=UniProt` and exact `referenceName` match), and **replace** the LLM's `ids` with the resolved set. If gene-name resolution returns nothing for a column, fall back to the LLM's accessions. Metabolite columns skip resolution: ChEBI accessions from the LLM are reliable when the dual-form rule is enforced. Rationale: open-weight models in the 100B–235B class (verified across `nvidia/nemotron-3-super-120b-a12b`, `openai/gpt-oss-120b`, `Qwen/Qwen3-235B-A22B`) reliably know gene symbols but mis-recall UniProt accessions, sometimes confidently emitting unrelated proteins. Reactome is the deterministic source of truth.
   - Per-ID Reactome validation: for each remaining ID call `ReactomeClient.validate_ids(ref)`; drop accessions that fail to resolve.
   - If *no* returned ID validates, set `reactome_validated: False` and `confidence = min(reported, 0.4)`. If at least one validates, mark `True`.
   - Drop unvalidated IDs from the `ids` list (keep validated ones only).

4. **Run on Sachs**, evaluate against gold:
   - Per-column accuracy: did the validated `ids` set match gold's `ids` set?
   - Iterate the prompt if Sachs accuracy < 0.9.
   - Once locked, commit `experiments/grounding_sachs.json`.

5. **Generalisation check (informal).** Run the *same* function with the *same* prompt on a small unseen dataset (e.g. a subset of DREAM4 PSN columns if available, otherwise a manually-constructed list of 5 unfamiliar phospho-protein names) to confirm the prompt is not overfit to Sachs phrasing. Document the result.

---

## Acceptance criteria

- [ ] Grounding accuracy on Sachs ≥ 0.9 vs `grounding/gold_sachs.json`.
- [ ] Every metabolite (PIP2, PIP3) returns ChEBI IDs, not UniProt or `null`.
- [ ] Every family (PKC, PKA, p44/42, pjnk) returns ≥ 2 UniProt IDs.
- [ ] No grounding entry has `reactome_validated: True` while all of its `ids` failed validation (logical consistency).
- [ ] Re-running the script hits the LLM cache for 100% of identical inputs.
- [ ] `task lint` and `task test` pass.

---

## Pitfalls

- **Don't trust the LLM to produce valid UniProt accessions.** It will sometimes hallucinate plausible-looking accessions. Always validate.
- **Validation must accept Complexes and EntitySets.** `get_entity_info` should return non-null when an ID maps to a Complex/EntitySet that contains the protein, not just a direct EntityWithAccessionedSequence.
- **Phospho prefix `p` is ambiguous.** `pAkt` could be Akt1, Akt2, or Akt3. Sachs uses Akt1 (UniProt P31749) by convention but the LLM may pick others. Few-shot examples should cover this explicitly.
- **Column-name conventions are dataset-specific.** Sachs columns may have idiosyncratic abbreviations (e.g. `pakts473` referencing Ser473 phosphorylation site). The prompt should accept these but also tolerate cleaner names from other datasets.
- **Confidence score asymmetry.** The LLM tends to over-report confidence. Down-weight aggressively on validation failure (cap at 0.4), and consider down-weighting moderately for partial matches (e.g. only 1 of 3 family members validates).

---

## Out of scope

- Causal reasoning over the grounded entities (Step 4).
- DREAM4 PSN gold curation (deferred to Step 7 unless data is available now).
- A deterministic rule-based grounder (parking lot).
