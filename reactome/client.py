"""Thin Reactome Content Service REST client with on-disk caching.

See ``reactome/endpoints.md`` for the list of REST endpoints used and
verified example responses. The client reuses ``llm.cache.cached_call`` so
every JSON response is replayable from disk; reruns with the network
disabled succeed when every requested payload already has a cache file.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx

from llm.cache import cached_call

REACTOME_BASE = "https://reactome.org/ContentService"
DEFAULT_CACHE_DIR = Path("cache/reactome")
HUMAN_SPECIES = "Homo sapiens"

Role = Literal["input", "output", "catalyst", "regulator"]
RegulationSign = Literal["positive", "negative", "unknown"]
EvidenceLayer = Literal["reaction", "co_pathway", "regulator_chain", "co_complex"]

_HIGHLIGHT_RE = re.compile(r"</?span[^>]*>")


def _strip_highlight(text: str | None) -> str:
    """Strip Reactome search ``<span class="highlighting">`` markup."""

    if not text:
        return ""
    return _HIGHLIGHT_RE.sub("", text)


def _normalize_chebi(identifier: str | None) -> str | None:
    """Reactome stores some ChEBI ids bare and some prefixed; normalise."""

    if not identifier:
        return None
    bare = identifier.upper().removeprefix("CHEBI:")
    return f"CHEBI:{bare}"


@dataclass(frozen=True)
class EntityRef:
    """A grounded variable reference: one or more accessions of a single kind."""

    kind: Literal["protein", "metabolite"]
    ids: list[str]
    display_name: str

    @property
    def normalised_ids(self) -> list[str]:
        if self.kind == "metabolite":
            return [n for n in (_normalize_chebi(i) for i in self.ids) if n]
        return [i for i in self.ids if i]


@dataclass(frozen=True)
class ReactionMeta:
    """Lightweight reaction metadata returned by the listing endpoints."""

    st_id: str
    display_name: str
    species: str | None = None


@dataclass(frozen=True)
class ReactionRecord:
    """A single (reaction × role-of-a × role-of-b) co-participation record."""

    reaction_id: str
    reaction_name: str
    pathway: str | None
    role_a: Role
    role_b: Role
    reaction_type: str
    sign: RegulationSign | None = None
    supporting_ids_a: tuple[str, ...] = field(default_factory=tuple)
    supporting_ids_b: tuple[str, ...] = field(default_factory=tuple)
    evidence_layer: EvidenceLayer = "reaction"


_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _fetch_with_retry(
    method: Callable[[], httpx.Response],
    *,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Run ``method`` with exponential backoff on transient 429 / 5xx."""

    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            response = method()
        except httpx.RequestError as exc:
            last_error = exc
            if attempt + 1 == max_attempts:
                raise
            sleep(base_delay * (2**attempt))
            continue

        if response.status_code in _RETRYABLE_STATUS and attempt + 1 < max_attempts:
            retry_after = response.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after is not None else 0.0
            except ValueError:
                delay = 0.0
            sleep(max(delay, base_delay * (2**attempt)))
            continue

        response.raise_for_status()
        return response.json()

    if last_error is not None:
        raise last_error
    raise RuntimeError("Reactome request retries exhausted without a response")


class ReactomeClient:
    """Cached, role-aware accessor for the Reactome Content Service REST API."""

    def __init__(
        self,
        *,
        base_url: str = REACTOME_BASE,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        timeout: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = base_url
        self.cache_dir = cache_dir
        self.timeout = timeout
        self._sleep = sleep
        self._memory_cache: dict[tuple[str, tuple[tuple[str, Any], ...]], Any] = {}

    def _get(self, path: str, **params: Any) -> Any:
        """Cached GET against the Reactome Content Service."""

        memory_key = (path, tuple(sorted(params.items())))
        if memory_key in self._memory_cache:
            return self._memory_cache[memory_key]

        payload = {"path": path, "params": params}

        def fetch() -> Any:
            def call() -> httpx.Response:
                return httpx.get(
                    f"{self.base_url}{path}",
                    params=params or None,
                    timeout=self.timeout,
                )

            return _fetch_with_retry(call, sleep=self._sleep)

        data = cached_call(self.cache_dir, payload, fetch)
        self._memory_cache[memory_key] = data
        return data

    # ------------------------------------------------------------------
    # Listing endpoints
    # ------------------------------------------------------------------

    def reactions_for_protein(self, uniprot: str) -> list[ReactionMeta]:
        """All Homo sapiens reactions involving ``uniprot`` directly or via a Complex."""

        try:
            data = self._get(f"/data/mapping/UniProt/{uniprot}/reactions")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return []
            raise

        return [
            ReactionMeta(
                st_id=item.get("stId") or item.get("stIdVersion") or "",
                display_name=item.get("displayName", ""),
                species=item.get("speciesName"),
            )
            for item in (data or [])
            if isinstance(item, dict)
            and item.get("speciesName") == HUMAN_SPECIES
            and item.get("stId")
        ]

    def reactions_for_metabolite(self, name: str) -> list[ReactionMeta]:
        """Search-derived reactions for a metabolite name (Homo sapiens only).

        The Reactome Content Service does not expose a stable ChEBI →
        reactions mapping (``/data/mapping/ChEBI/{id}/reactions`` 404s for
        PIP2/PIP3). The ``All-species`` canonical SimpleEntity returned by
        ``/data/query/enhanced/{R-ALL-...}`` has empty ``consumedByEvent`` /
        ``producedByEvent`` fields.

        The reliable workaround is to query the search index directly for
        reactions whose name mentions ``name``, restrict to Homo sapiens, and
        let downstream role assignment use ``reaction_participants`` to
        verify the metabolite's ChEBI accession is actually a refEntity.
        """

        result = self._get(
            "/search/query",
            query=name,
            species=HUMAN_SPECIES,
            types="Reaction",
        )
        metas: list[ReactionMeta] = []
        seen: set[str] = set()
        for bucket in result.get("results", []) or []:
            if bucket.get("typeName") != "Reaction":
                continue
            for entry in bucket.get("entries", []) or []:
                st_id = entry.get("stId")
                if not st_id or st_id in seen:
                    continue
                species_list = entry.get("species") or []
                if HUMAN_SPECIES not in species_list:
                    continue
                seen.add(st_id)
                metas.append(
                    ReactionMeta(
                        st_id=st_id,
                        display_name=_strip_highlight(entry.get("name", "")),
                        species=HUMAN_SPECIES,
                    )
                )
        return metas

    # ------------------------------------------------------------------
    # Reaction inspection endpoints
    # ------------------------------------------------------------------

    def reaction_details(self, rxn_st_id: str) -> dict[str, Any]:
        """Full reaction object with ``input`` / ``output`` / ``catalystActivity`` /
        ``regulatedBy`` participants returned inline."""

        return self._get(f"/data/query/enhanced/{rxn_st_id}")

    def discover(self, st_id: str) -> dict[str, Any]:
        """Discover endpoint payload for a Reactome stable identifier."""

        return self._get(f"/data/discover/{st_id}")

    def reaction_participants(self, rxn_st_id: str) -> dict[int, list[dict[str, Any]]]:
        """Per-PhysicalEntity flattened reference entities for the given reaction.

        Returns ``{peDbId: [refEntity, ...]}`` so callers can ask "for this PE
        in the reaction, which UniProt / ChEBI accessions are visible (after
        Complex / EntitySet expansion)?"
        """

        data = self._get(f"/data/participants/{rxn_st_id}")
        flattened: dict[int, list[dict[str, Any]]] = {}
        for entry in data or []:
            if not isinstance(entry, dict):
                continue
            pe_db_id = entry.get("peDbId")
            if pe_db_id is None:
                continue
            refs = [
                r for r in entry.get("refEntities", []) or [] if isinstance(r, dict)
            ]
            flattened[int(pe_db_id)] = refs
        return flattened

    # ------------------------------------------------------------------
    # Co-participation / role assignment
    # ------------------------------------------------------------------

    def get_reaction_context(self, a: EntityRef, b: EntityRef) -> list[ReactionRecord]:
        """Find reactions in which ``a`` and ``b`` co-participate, role-typed."""

        reactions_a = self._reactions_for_entity(a)
        reactions_b = self._reactions_for_entity(b)
        shared = sorted(
            {meta.st_id for meta in reactions_a} & {meta.st_id for meta in reactions_b}
        )

        meta_by_id: dict[str, ReactionMeta] = {}
        for meta in (*reactions_a, *reactions_b):
            meta_by_id.setdefault(meta.st_id, meta)

        records: list[ReactionRecord] = []
        for st_id in shared:
            try:
                details = self.reaction_details(st_id)
                participants = self.reaction_participants(st_id)
            except httpx.HTTPStatusError:
                continue

            roles_a = self._roles_in_reaction(a, details, participants)
            roles_b = self._roles_in_reaction(b, details, participants)
            if not roles_a or not roles_b:
                continue

            reaction_type = (
                details.get("schemaClass") or details.get("className") or "Reaction"
            )
            reaction_name = (
                details.get("displayName")
                or meta_by_id.get(st_id, ReactionMeta(st_id, "")).display_name
            )
            pathway = self._pick_pathway(details)

            for role_a, sign_a, hits_a in roles_a:
                for role_b, sign_b, hits_b in roles_b:
                    if a == b and role_a == role_b:
                        continue
                    sign = (
                        sign_a
                        if role_a == "regulator"
                        else sign_b
                        if role_b == "regulator"
                        else None
                    )
                    records.append(
                        ReactionRecord(
                            reaction_id=st_id,
                            reaction_name=reaction_name,
                            pathway=pathway,
                            role_a=role_a,
                            role_b=role_b,
                            reaction_type=reaction_type,
                            sign=sign,
                            supporting_ids_a=tuple(sorted(hits_a)),
                            supporting_ids_b=tuple(sorted(hits_b)),
                        )
                    )
        return records

    def pathways_for_entity(self, entity: EntityRef) -> set[str]:
        """Reactome pathway stable ids referenced by reactions involving ``entity``."""

        return set(self._pathways_for_entity_with_names(entity))

    def co_pathway_records(self, a: EntityRef, b: EntityRef) -> list[ReactionRecord]:
        """Pathway-level evidence for entities that share Reactome pathways."""

        pathways_a = self._pathways_for_entity_with_names(a)
        pathways_b = self._pathways_for_entity_with_names(b)
        records: list[ReactionRecord] = []

        for st_id in sorted(set(pathways_a) & set(pathways_b)):
            pathway_name = pathways_a.get(st_id) or pathways_b.get(st_id) or st_id
            if pathway_name == st_id:
                try:
                    discovered = self.discover(st_id)
                except httpx.HTTPStatusError:
                    discovered = {}
                if isinstance(discovered, dict):
                    pathway_name = (
                        _strip_highlight(discovered.get("displayName")) or st_id
                    )
            records.append(
                ReactionRecord(
                    reaction_id=st_id,
                    reaction_name=pathway_name,
                    pathway=pathway_name,
                    role_a="input",
                    role_b="input",
                    reaction_type="Pathway",
                    sign=None,
                    supporting_ids_a=tuple(a.normalised_ids),
                    supporting_ids_b=tuple(b.normalised_ids),
                    evidence_layer="co_pathway",
                )
            )
        return records

    def regulator_chain_records(
        self, a: EntityRef, b: EntityRef
    ) -> list[ReactionRecord]:
        """One-hop evidence where ``a`` regulates or catalyses reactions producing ``b``."""

        reaction_keys = {
            (record.reaction_id, record.role_a, record.role_b, record.sign)
            for record in self.get_reaction_context(a, b)
        }
        records: list[ReactionRecord] = []
        seen: set[tuple[str, Role, Role, RegulationSign | None]] = set()
        accessions_a = set(a.normalised_ids)
        accessions_b = set(b.normalised_ids)

        for meta in self._reactions_for_entity(b):
            try:
                details = self.reaction_details(meta.st_id)
                participants = self.reaction_participants(meta.st_id)
            except httpx.HTTPStatusError:
                continue

            b_hits = self._output_hits(b, details, participants, accessions_b)
            if not b_hits:
                continue

            reaction_type = (
                details.get("schemaClass") or details.get("className") or "Reaction"
            )
            reaction_name = details.get("displayName") or meta.display_name
            pathway = self._pick_pathway(details)

            for reg in details.get("regulatedBy", []) or []:
                resolved = self._resolve_regulation(reg)
                if resolved is None:
                    continue
                regulator_pe, sign = resolved
                hits_a = self._matches_for_pe(
                    regulator_pe, participants, accessions_a, a.kind == "metabolite"
                )
                if hits_a:
                    self._append_chain_record(
                        records,
                        seen,
                        reaction_keys,
                        meta.st_id,
                        reaction_name,
                        pathway,
                        reaction_type,
                        "regulator",
                        sign,
                        hits_a,
                        b_hits,
                    )

            for ca in details.get("catalystActivity", []) or []:
                if not isinstance(ca, dict):
                    continue
                hits_a = self._matches_for_pe(
                    ca.get("physicalEntity"),
                    participants,
                    accessions_a,
                    a.kind == "metabolite",
                )
                if hits_a:
                    self._append_chain_record(
                        records,
                        seen,
                        reaction_keys,
                        meta.st_id,
                        reaction_name,
                        pathway,
                        reaction_type,
                        "catalyst",
                        None,
                        hits_a,
                        b_hits,
                    )

        return records

    def co_complex_records(self, a: EntityRef, b: EntityRef) -> list[ReactionRecord]:
        """Complex-membership evidence mined from cached participant expansions."""

        records: list[ReactionRecord] = []
        seen_complexes: set[str] = set()
        accessions_a = set(a.normalised_ids)
        accessions_b = set(b.normalised_ids)
        is_metabolite_a = a.kind == "metabolite"
        is_metabolite_b = b.kind == "metabolite"

        for meta in self._participant_index_metas(a, b):
            try:
                raw_participants = self._get(f"/data/participants/{meta.st_id}")
            except httpx.HTTPStatusError:
                continue
            for entry in raw_participants or []:
                if not isinstance(entry, dict):
                    continue
                pe_db_id = entry.get("peDbId")
                if pe_db_id is None:
                    continue
                schema = entry.get("schemaClass") or ""
                if schema and schema != "Complex":
                    continue
                refs = [
                    ref
                    for ref in entry.get("refEntities", []) or []
                    if isinstance(ref, dict)
                ]
                hits_a = self._matching_ref_ids(refs, accessions_a, is_metabolite_a)
                hits_b = self._matching_ref_ids(refs, accessions_b, is_metabolite_b)
                if not hits_a or not hits_b:
                    continue

                complex_id = str(pe_db_id)
                complex_name = entry.get("displayName") or complex_id
                try:
                    enhanced = self._get(f"/data/query/enhanced/{pe_db_id}")
                except httpx.HTTPStatusError:
                    enhanced = {}
                if isinstance(enhanced, dict):
                    complex_id = enhanced.get("stId") or complex_id
                    complex_name = enhanced.get("displayName") or complex_name

                if complex_id in seen_complexes:
                    continue
                seen_complexes.add(complex_id)
                records.append(
                    ReactionRecord(
                        reaction_id=complex_id,
                        reaction_name=_strip_highlight(complex_name),
                        pathway=None,
                        role_a="input",
                        role_b="input",
                        reaction_type="ComplexMembership",
                        sign=None,
                        supporting_ids_a=tuple(sorted(hits_a)),
                        supporting_ids_b=tuple(sorted(hits_b)),
                        evidence_layer="co_complex",
                    )
                )
        return records

    def get_evidence_records(
        self,
        a: EntityRef,
        b: EntityRef,
        *,
        layers: tuple[str, ...] = (
            "reaction",
            "co_pathway",
            "regulator_chain",
            "co_complex",
        ),
    ) -> list[ReactionRecord]:
        """Return additive Reactome evidence records in caller-requested layer order."""

        records: list[ReactionRecord] = []
        for layer in layers:
            if layer == "reaction":
                records.extend(self.get_reaction_context(a, b))
            elif layer == "co_pathway":
                records.extend(self.co_pathway_records(a, b))
            elif layer == "regulator_chain":
                records.extend(self.regulator_chain_records(a, b))
            elif layer == "co_complex":
                records.extend(self.co_complex_records(a, b))
            else:
                raise ValueError(f"Unsupported Reactome evidence layer: {layer}")
        return records

    def _reactions_for_entity(self, entity: EntityRef) -> list[ReactionMeta]:
        merged: dict[str, ReactionMeta] = {}
        if entity.kind == "protein":
            for accession in entity.normalised_ids:
                for meta in self.reactions_for_protein(accession):
                    merged.setdefault(meta.st_id, meta)
        else:
            for meta in self.reactions_for_metabolite(entity.display_name):
                merged.setdefault(meta.st_id, meta)
        return list(merged.values())

    def _pathways_for_entity_with_names(self, entity: EntityRef) -> dict[str, str]:
        pathways: dict[str, str] = {}
        for meta in self._reactions_for_entity(entity):
            try:
                details = self.reaction_details(meta.st_id)
            except httpx.HTTPStatusError:
                continue
            for entry in details.get("eventOf", []) or []:
                if not isinstance(entry, dict):
                    continue
                schema = entry.get("schemaClass") or entry.get("className") or ""
                st_id = entry.get("stId")
                if not st_id:
                    continue
                if "pathway" not in schema.lower() and schema:
                    continue
                display_name = _strip_highlight(entry.get("displayName")) or st_id
                pathways.setdefault(st_id, display_name)
        return pathways

    def _output_hits(
        self,
        entity: EntityRef,
        details: dict[str, Any],
        participants: dict[int, list[dict[str, Any]]],
        accessions: set[str],
    ) -> set[str]:
        hits: set[str] = set()
        for pe in details.get("output", []) or []:
            hits.update(
                self._matches_for_pe(
                    pe,
                    participants,
                    accessions,
                    entity.kind == "metabolite",
                )
            )
        return hits

    def _append_chain_record(
        self,
        records: list[ReactionRecord],
        seen: set[tuple[str, Role, Role, RegulationSign | None]],
        reaction_keys: set[tuple[str, Role, Role, RegulationSign | None]],
        reaction_id: str,
        reaction_name: str,
        pathway: str | None,
        reaction_type: str,
        role_a: Literal["catalyst", "regulator"],
        sign: RegulationSign | None,
        hits_a: set[str],
        hits_b: set[str],
    ) -> None:
        key = (reaction_id, role_a, "output", sign)
        if key in seen or key in reaction_keys:
            return
        seen.add(key)
        records.append(
            ReactionRecord(
                reaction_id=reaction_id,
                reaction_name=reaction_name,
                pathway=pathway,
                role_a=role_a,
                role_b="output",
                reaction_type=reaction_type,
                sign=sign,
                supporting_ids_a=tuple(sorted(hits_a)),
                supporting_ids_b=tuple(sorted(hits_b)),
                evidence_layer="regulator_chain",
            )
        )

    def _participant_index_metas(
        self, a: EntityRef, b: EntityRef
    ) -> list[ReactionMeta]:
        merged: dict[str, ReactionMeta] = {}
        for meta in (*self._reactions_for_entity(a), *self._reactions_for_entity(b)):
            merged.setdefault(meta.st_id, meta)
        return list(merged.values())

    def _roles_in_reaction(
        self,
        entity: EntityRef,
        details: dict[str, Any],
        participants: dict[int, list[dict[str, Any]]],
    ) -> list[tuple[Role, RegulationSign | None, set[str]]]:
        accessions = set(entity.normalised_ids)
        is_metabolite = entity.kind == "metabolite"
        roles: list[tuple[Role, RegulationSign | None, set[str]]] = []

        for role, key in (("input", "input"), ("output", "output")):
            for pe in details.get(key, []) or []:
                hits = self._matches_for_pe(pe, participants, accessions, is_metabolite)
                if hits:
                    roles.append((role, None, hits))

        for ca in details.get("catalystActivity", []) or []:
            if not isinstance(ca, dict):
                continue
            pe = ca.get("physicalEntity")
            hits = self._matches_for_pe(pe, participants, accessions, is_metabolite)
            if hits:
                roles.append(("catalyst", None, hits))

        for reg in details.get("regulatedBy", []) or []:
            resolved = self._resolve_regulation(reg)
            if resolved is None:
                continue
            regulator_pe, sign = resolved
            hits = self._matches_for_pe(
                regulator_pe, participants, accessions, is_metabolite
            )
            if hits:
                roles.append(("regulator", sign, hits))

        return self._dedupe_roles(roles)

    @staticmethod
    def _dedupe_roles(
        roles: Iterable[tuple[Role, RegulationSign | None, set[str]]],
    ) -> list[tuple[Role, RegulationSign | None, set[str]]]:
        seen: dict[tuple[Role, RegulationSign | None, frozenset[str]], None] = {}
        for role, sign, hits in roles:
            seen.setdefault((role, sign, frozenset(hits)), None)
        return [(role, sign, set(hits)) for role, sign, hits in seen]

    def _resolve_regulation(
        self, reg: Any
    ) -> tuple[dict[str, Any], RegulationSign] | None:
        if isinstance(reg, int):
            try:
                reg = self._get(f"/data/query/enhanced/{reg}")
            except httpx.HTTPStatusError:
                return None
        if not isinstance(reg, dict):
            return None
        regulator = reg.get("regulator")
        if not isinstance(regulator, dict):
            return None
        schema = (reg.get("schemaClass") or reg.get("className") or "").lower()
        if "negative" in schema:
            sign: RegulationSign = "negative"
        elif "positive" in schema:
            sign = "positive"
        else:
            sign = "unknown"
        return regulator, sign

    def _matches_for_pe(
        self,
        pe: Any,
        participants: dict[int, list[dict[str, Any]]],
        accessions: set[str],
        is_metabolite: bool,
    ) -> set[str]:
        if not isinstance(pe, dict):
            return set()
        db_id = pe.get("dbId")
        if db_id is None:
            return set()
        refs = participants.get(int(db_id), [])
        hits: set[str] = set()
        for ref in refs:
            identifier = ref.get("identifier")
            if not identifier:
                continue
            if is_metabolite:
                normalised = _normalize_chebi(identifier)
                if normalised and normalised in accessions:
                    hits.add(normalised)
            else:
                if identifier in accessions:
                    hits.add(identifier)
        return hits

    @staticmethod
    def _matching_ref_ids(
        refs: Iterable[dict[str, Any]],
        accessions: set[str],
        is_metabolite: bool,
    ) -> set[str]:
        hits: set[str] = set()
        for ref in refs:
            identifier = ref.get("identifier")
            if not identifier:
                continue
            if is_metabolite:
                normalised = _normalize_chebi(identifier)
                if normalised and normalised in accessions:
                    hits.add(normalised)
            elif identifier in accessions:
                hits.add(identifier)
        return hits

    @staticmethod
    def _pick_pathway(details: dict[str, Any]) -> str | None:
        for field_name in ("eventOf", "inferredFrom"):
            container = details.get(field_name)
            if isinstance(container, list) and container:
                first = container[0]
                if isinstance(first, dict) and first.get("displayName"):
                    return _strip_highlight(first["displayName"])
        return None


# ----------------------------------------------------------------------
# LLM-facing formatting
# ----------------------------------------------------------------------

_ROLE_PHRASE: dict[Role, str] = {
    "input": "is an input",
    "output": "is an output",
    "catalyst": "acts in the catalyst complex",
    "regulator": "acts as a regulator",
}


def _role_phrase(role: Role, sign: RegulationSign | None) -> str:
    base = _ROLE_PHRASE[role]
    if role == "regulator" and sign in {"positive", "negative"}:
        return f"acts as a {sign} regulator"
    return base


def format_context_for_llm(
    records: list[ReactionRecord], name_a: str, name_b: str
) -> str:
    """Render a natural-language paragraph per reaction record for prompt inclusion."""

    if not records:
        return f"No Reactome reactions co-involve {name_a} and {name_b}."

    lines: list[str] = []
    for record in records:
        if record.evidence_layer == "co_pathway":
            lines.append(
                f"{name_a} and {name_b} both participate in the Reactome pathway "
                f"'{record.pathway}' (stId {record.reaction_id})."
            )
            continue
        if record.evidence_layer == "regulator_chain":
            product_phrase = (
                f"{name_a} {_role_phrase(record.role_a, record.sign)} a reaction "
                f"whose product includes {name_b}"
            )
            lines.append(
                f"{product_phrase} (terminal reaction {record.reaction_id}, "
                f"'{record.reaction_name}')."
            )
            continue
        if record.evidence_layer == "co_complex":
            lines.append(
                f"{name_a} and {name_b} co-occur as members of the Reactome Complex "
                f"'{record.reaction_name}' (stId {record.reaction_id}), without a "
                f"shared reaction."
            )
            continue
        pathway = (
            f"In the {record.pathway} pathway, " if record.pathway else "In Reactome, "
        )
        lines.append(
            f"{pathway}{name_a} {_role_phrase(record.role_a, record.sign)} in reaction "
            f"{record.reaction_id} ('{record.reaction_name}'), where {name_b} "
            f"{_role_phrase(record.role_b, record.sign)}. "
            f"Reaction type: {record.reaction_type}."
        )
    return "\n".join(lines)
