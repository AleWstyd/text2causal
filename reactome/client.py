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

    def _get(self, path: str, **params: Any) -> Any:
        """Cached GET against the Reactome Content Service."""

        payload = {"path": path, "params": params}

        def fetch() -> Any:
            def call() -> httpx.Response:
                return httpx.get(
                    f"{self.base_url}{path}",
                    params=params or None,
                    timeout=self.timeout,
                )

            return _fetch_with_retry(call, sleep=self._sleep)

        return cached_call(self.cache_dir, payload, fetch)

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
