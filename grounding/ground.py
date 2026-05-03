"""Batched LLM variable grounding + post-hoc Reactome validation."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from openai import OpenAI

from llm.cache import cached_call
from llm.client import MODEL, OPENROUTER_BASE_URL
from reactome.client import EntityRef, ReactomeClient, _normalize_chebi

CHEBI_PAIR_PIP2: frozenset[str] = frozenset({"CHEBI:18348", "CHEBI:58456"})
CHEBI_PAIR_PIP3: frozenset[str] = frozenset({"CHEBI:16618", "CHEBI:57836"})

GROUNDING_SYSTEM = """You are a bioinformatics expert in proteomics and cell signalling. Your task is
to map dataset column names to canonical biomedical identifiers.

Rules:
- Prefix 'p' typically denotes a phosphorylated measurement (e.g. praf = phospho-RAF1).
- Return human (Homo sapiens) UniProt IDs for proteins, ChEBI IDs for metabolites/lipids.
- For protein families measured by pan-isoform antibodies (e.g. PKC, PKA, ERK1/2),
  return a LIST of canonical UniProt IDs covering the major isoforms; a "family" must
  contain ≥2 distinct gene products with established gene→UniProt mappings.
- For phosphoinositide metabolites PIP2 and PIP3, you MUST include BOTH protonated AND
  deprotonated ChEBI forms in `ids`:
  • PIP2: CHEBI:18348 and CHEBI:58456
  • PIP3: CHEBI:16618 and CHEBI:57836
  Omitting either form halves Reactome metabolite hits and must be avoided.
- Never invent UniProt accessions. UniProts are NOT sequential — P17612 is PRKACA,
  but P17613, P17614, P17615 are unrelated proteins. If unsure of a specific
  accession, drop it; do not extrapolate. Use only well-established gene→UniProt
  mappings drawn from your training knowledge.
- Report a confidence (0.0-1.0) per column.
- If you cannot map with confidence > 0.5, return null for `ids`.

Few-shot anchors (illustrative of the mapping style expected):
• Single phosphoprotein → `"kind":"protein"`, `"ids":["<UniProt>"]`.
• Pan-family kinase → `"kind":"family"`, `"ids":[<≥2 UniProts of the family>]`.
• Phosphoinositide lipid → `"kind":"metabolite"`, `"ids":[<both ChEBI forms>]`.

Do not claim validation against databases; return only identifiers and reasoning.

Output ONLY a JSON object. No preamble, no markdown."""

GROUNDING_USER_TEMPLATE = """Dataset: {dataset_description}
Domain: {domain_hint}
Columns: {column_list}

Return a JSON object with this exact schema:
{{
  "<column_name>": {{
    "kind": "protein" | "metabolite" | "family",
    "ids": ["<UniProt or ChEBI accession>", ...] | null,
    "canonical_name": "<human-readable name>",
    "gene_names": ["<HGNC gene symbol>", ...],
    "confidence": <float 0.0-1.0>,
    "reasoning": "<one sentence explaining the mapping>"
  }}
}}"""

# UniProtKB accession (strict enough for Sachs / kinase ids).
_UNIPROT_RE = re.compile(
    r"^(?:UniProt\s*:|SP\s*:|TR\s*:)?(?P<id>[OPQ][0-9][A-Z0-9]{3}[0-9]|"
    r"[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})$",
    re.IGNORECASE,
)


def unwrap_json_llm_response(raw: str) -> str:
    """Strip optional ``` / ```json fences; models often wrap JSON."""

    s = raw.strip()
    if "```" not in s:
        return s
    start = s.find("```")
    rest = s[start + 3 :]
    if rest.lower().startswith("json"):
        rest = rest[4:].lstrip()
    else:
        rest = rest.lstrip()
    if rest.rstrip().endswith("```"):
        rest = rest.rstrip()[:-3]
    return rest.strip()


def parse_grounding_response(content: str) -> dict[str, Any]:
    to_parse = unwrap_json_llm_response(content)
    try:
        parsed = json.loads(to_parse)
    except json.JSONDecodeError as exc:
        msg = f"Grounding response is not valid JSON: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(parsed, dict):
        msg = "Grounding JSON root must be an object keyed by column name."
        raise TypeError(msg)
    return parsed


def build_grounding_messages(
    *,
    dataset_description: str,
    domain_hint: str,
    columns: list[str],
) -> list[dict[str, str]]:
    column_list = json.dumps(columns)
    user = GROUNDING_USER_TEMPLATE.format(
        dataset_description=dataset_description,
        domain_hint=domain_hint,
        column_list=column_list,
    )
    return [
        {"role": "system", "content": GROUNDING_SYSTEM},
        {"role": "user", "content": user},
    ]


def _normalise_protein_identifier(token: str) -> str | None:
    m = _UNIPROT_RE.match(token.strip())
    if not m:
        return None
    return m.group("id").upper()


def _normalise_metabolite_identifier(token: str) -> str | None:
    return _normalize_chebi(token.strip())


def normalize_id_list(
    kind: Literal["protein", "metabolite", "family"], raw_ids: object
) -> list[str]:
    if raw_ids is None:
        return []
    if not isinstance(raw_ids, list):
        return []
    mapper = (
        _normalise_protein_identifier
        if kind in {"protein", "family"}
        else _normalise_metabolite_identifier
    )
    seen: set[str] = set()
    out: list[str] = []
    for item in raw_ids:
        if not isinstance(item, str):
            continue
        norm = mapper(item)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def _openrouter_client() -> OpenAI:
    key = os.environ.get("OPEN_ROUTER_API_KEY")
    if not key:
        msg = "OPEN_ROUTER_API_KEY is not set (e.g. in .env for `task run`)."
        raise ValueError(msg)
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)


def default_llm_fetch(
    messages: list[dict[str, Any]], *, cache_dir: Path, model: str = MODEL
) -> dict[str, Any]:
    payload = {"task": "grounding", "model": model, "messages": messages}

    def fetch() -> dict[str, Any]:
        response = _openrouter_client().chat.completions.create(
            model=model,
            messages=messages,
        )
        return response.model_dump()

    return cached_call(cache_dir, payload, fetch)


def _entity_ref_from_entry(
    kind_raw: object,
    ids: list[str],
    canonical_name: str,
    *,
    fallback_name: str,
) -> EntityRef | None:
    if not ids:
        return None
    if kind_raw == "metabolite":
        rkind = "metabolite"
        # For metabolites, validation goes through Reactome's name-based
        # search; the LLM's canonical_name (e.g. "Phosphatidylinositol
        # 4,5-bisphosphate") is freeform and triggers slow live searches.
        # The dataset column name (e.g. "PIP2") is cache-friendly and
        # already populated from Step 2.
        name = fallback_name
    else:
        rkind = "protein"
        name = canonical_name.strip() if canonical_name else fallback_name
    return EntityRef(kind=rkind, ids=ids, display_name=name)


def served_model_from_response(response: dict[str, Any]) -> str | None:
    raw = response.get("model")
    return raw if isinstance(raw, str) else None


@dataclass(frozen=True)
class Grounding:
    column: str
    kind: Literal["protein", "metabolite", "family"]
    ids: list[str]
    canonical_name: str
    gene_names: list[str]
    confidence: float
    reasoning: str
    reactome_validated: bool
    served_model: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def ground_columns(
    columns: list[str],
    dataset_description: str,
    domain_hint: str,
    *,
    llm_fetch: Callable[[list[dict[str, Any]], Path], dict[str, Any]] | None = None,
    reactome_client: ReactomeClient | None = None,
    cache_dir: Path = Path("cache/llm"),
) -> dict[str, Grounding]:
    messages = build_grounding_messages(
        dataset_description=dataset_description,
        domain_hint=domain_hint,
        columns=list(columns),
    )
    cwd = Path(cache_dir)

    def _default_fetch(m: list[dict[str, Any]], c: Path) -> dict[str, Any]:
        return default_llm_fetch(m, cache_dir=c)

    fetch_impl = llm_fetch or _default_fetch
    response = fetch_impl(messages, cwd)
    served = served_model_from_response(response)
    choices = response.get("choices") or []
    if not isinstance(choices, list) or not choices:
        msg = "Grounding LLM response has no choices."
        raise ValueError(msg)
    first = choices[0]
    if not isinstance(first, dict):
        msg = "Malformed grounding LLM choice."
        raise ValueError(msg)
    message_obj = first.get("message") or {}
    content = ""
    if isinstance(message_obj, dict):
        raw_c = message_obj.get("content")
        content = raw_c.strip() if isinstance(raw_c, str) else ""

    parsed = parse_grounding_response(content)
    requested = columns
    parsed_keys = frozenset(parsed.keys())
    required = frozenset(requested)
    if missing_cols := sorted(required - parsed_keys):
        msg = (
            "Grounding JSON missing columns: "
            f"{missing_cols}. Got keys={sorted(parsed_keys)!r}"
        )
        raise ValueError(msg)

    client = reactome_client or ReactomeClient()
    out: dict[str, Grounding] = {}

    for col in requested:
        entry = parsed[col]
        if not isinstance(entry, dict):
            msg = f"Grounding entry for {col} must be an object."
            raise TypeError(msg)

        raw_kind = entry.get("kind")
        if raw_kind not in {"protein", "metabolite", "family"}:
            raise ValueError(
                f'Column "{col}": invalid or missing kind in LLM grounding output.'
            )
        kind_lit: Literal["protein", "metabolite", "family"] = raw_kind

        canonical = ""
        gn_raw = entry.get("canonical_name")
        if isinstance(gn_raw, str):
            canonical = gn_raw.strip()
        reasoning = ""
        r_raw = entry.get("reasoning")
        if isinstance(r_raw, str):
            reasoning = r_raw.strip()
        genes: list[str] = []
        g_raw = entry.get("gene_names")
        if isinstance(g_raw, list):
            genes = [x.strip() for x in g_raw if isinstance(x, str) and x.strip()]

        ids = normalize_id_list(kind_lit, entry.get("ids"))

        confidence = 0.0
        c_raw = entry.get("confidence")
        if isinstance(c_raw, (int, float)) and not isinstance(c_raw, bool):
            confidence = float(c_raw)

        reported = confidence
        ref = _entity_ref_from_entry(
            kind_lit,
            ids,
            canonical,
            fallback_name=col,
        )

        validated = False
        if ref is not None and client.get_entity_info(ref) is not None:
            validated = True
        final_conf = reported if validated else min(reported, 0.4)

        out[col] = Grounding(
            column=col,
            kind=kind_lit,
            ids=ids,
            canonical_name=canonical,
            gene_names=genes,
            confidence=final_conf,
            reasoning=reasoning,
            reactome_validated=validated,
            served_model=served,
        )

    return out
