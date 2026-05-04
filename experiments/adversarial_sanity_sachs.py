"""Adversarial sanity checks for Sachs Reactome-context reasoning."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Final

from constraints.constraint_builder import ClaimRecord
from experiments.constraint_quality import compute_quality
from grounding.ground import Grounding
from reactome.client import EntityRef, ReactomeClient, format_context_for_llm
from reasoning.reason import (
    CausalClaim,
    aggregate_passes,
    build_reason_messages,
    default_llm_fetch,
    parse_claim_response,
)
from utils.load_data import load_sachs_dataset

GROUNDING_PATH: Final[Path] = Path("experiments/grounding_sachs.json")
OUTPUT_PATH: Final[Path] = Path("experiments/adversarial_sanity_sachs.json")
CACHE_DIR: Final[Path] = Path("cache/llm_adversarial")
ACCESSION_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9])\b"
)
SACHS_ACCESSIONS: Final[tuple[str, ...]] = (
    "P15056",
    "Q02750",
    "P27361",
    "P28482",
    "P19174",
    "P17252",
    "P31749",
    "Q16539",
)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def _groundings() -> dict[str, Grounding]:
    blob = json.loads(GROUNDING_PATH.read_text(encoding="utf-8"))
    raw = blob["predicted"]
    return {
        column: Grounding(
            column=str(entry.get("column", column)),
            kind=entry["kind"],
            ids=list(entry.get("ids") or []),
            canonical_name=str(entry.get("canonical_name", "")),
            gene_names=list(entry.get("gene_names") or []),
            confidence=float(entry.get("confidence", 0.0)),
            reasoning=str(entry.get("reasoning", "")),
            reactome_validated=bool(entry.get("reactome_validated", False)),
            served_model=entry.get("served_model"),
        )
        for column, entry in raw.items()
    }


def _entity_ref(column: str, grounding: Grounding) -> EntityRef:
    kind = "metabolite" if grounding.kind == "metabolite" else "protein"
    return EntityRef(kind=kind, ids=list(grounding.ids), display_name=column)


def _shuffle_accessions(context: str) -> str:
    hits = ACCESSION_RE.findall(context)
    if not hits:
        return context + "\n\n[Adversarial note: no accession tokens found to shuffle.]"
    mapping: dict[str, str] = {}
    for i, accession in enumerate(sorted(set(hits))):
        mapping[accession] = SACHS_ACCESSIONS[(i + 3) % len(SACHS_ACCESSIONS)]

    def _replace(match: re.Match[str]) -> str:
        return mapping.get(match.group(0), match.group(0))

    return ACCESSION_RE.sub(_replace, context)


def _wrong_protein_context(client: ReactomeClient) -> str:
    f2 = EntityRef(kind="protein", ids=["P00734"], display_name="coagulation factor II")
    fga = EntityRef(
        kind="protein", ids=["P02671"], display_name="fibrinogen alpha chain"
    )
    records = client.get_evidence_records(f2, fga)
    if not records:
        return (
            "Unrelated context: coagulation factor II and fibrinogen are blood "
            "coagulation proteins, not members of the Sachs MAPK/PI3K signalling panel."
        )
    return format_context_for_llm(
        records, "coagulation factor II", "fibrinogen alpha chain"
    )


def _no_context_claim(var_a: str, var_b: str) -> CausalClaim:
    return CausalClaim(
        var_a=var_a,
        var_b=var_b,
        cause="unknown",
        effect="unknown",
        confidence=0.0,
        constraint_type="no_context",
        supporting_reactions=[],
        contradicting_reactions=[],
        reasoning="No Reactome evidence after adversarial setup.",
        reactome_context_size=0,
        served_model=None,
    )


def _unknown_response_claim(
    var_a: str,
    var_b: str,
    *,
    served_model: str | None,
    reason: str,
    context_size: int,
) -> CausalClaim:
    return CausalClaim(
        var_a=var_a,
        var_b=var_b,
        cause="unknown",
        effect="unknown",
        confidence=0.0,
        constraint_type="unknown",
        supporting_reactions=[],
        contradicting_reactions=[],
        reasoning=reason,
        reactome_context_size=context_size,
        served_model=served_model,
        notes=["adversarial_response_missing_content"],
    )


def _reason_with_context_text(
    *,
    var_a: str,
    var_b: str,
    entity_a: EntityRef,
    entity_b: EntityRef,
    context_text: str,
    context_size: int,
    vocabulary: list[str],
) -> CausalClaim:
    messages = build_reason_messages(
        name_a=var_a,
        name_b=var_b,
        ids_a=list(entity_a.ids),
        ids_b=list(entity_b.ids),
        formatted_context=context_text,
        vocabulary=vocabulary,
    )
    response = default_llm_fetch(messages, cache_dir=CACHE_DIR)
    served = response.get("model") if isinstance(response, dict) else None
    if not isinstance(served, str):
        served = None
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise ValueError(f"Malformed adversarial response for {var_a}->{var_b}")
    message = choices[0].get("message") or {}
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        return _unknown_response_claim(
            var_a,
            var_b,
            served_model=served,
            reason="Adversarial LLM response did not contain parseable content.",
            context_size=context_size,
        )
    return parse_claim_response(
        content,
        var_a=var_a,
        var_b=var_b,
        vocabulary=vocabulary,
        context_size=context_size,
        served_model=served,
        supporting_pool=set(),
    )


def _run_variant(variant: str, client: ReactomeClient) -> list[dict[str, Any]]:
    grounding = _groundings()
    vocabulary = sorted(grounding)
    wrong_context = (
        _wrong_protein_context(client) if variant == "wrong_protein" else None
    )
    claims: list[dict[str, Any]] = []

    for var_a in vocabulary:
        for var_b in vocabulary:
            if var_a == var_b:
                continue
            a_ref = _entity_ref(var_a, grounding[var_a])
            b_ref = _entity_ref(var_b, grounding[var_b])
            records = client.get_evidence_records(a_ref, b_ref)
            if not records:
                agg = _no_context_claim(var_a, var_b)
                claims.append(agg.as_dict())
                continue

            base_context = format_context_for_llm(records, var_a, var_b)
            if variant == "shuffled_accession":
                context = _shuffle_accessions(base_context)
            elif variant == "wrong_protein":
                assert wrong_context is not None
                context = wrong_context
            else:
                raise ValueError(f"Unknown adversarial variant: {variant}")

            forward = _reason_with_context_text(
                var_a=var_a,
                var_b=var_b,
                entity_a=a_ref,
                entity_b=b_ref,
                context_text=context,
                context_size=len(records),
                vocabulary=vocabulary,
            )
            reverse_records = client.get_evidence_records(b_ref, a_ref)
            reverse_context = format_context_for_llm(reverse_records, var_b, var_a)
            if variant == "shuffled_accession":
                reverse_context = _shuffle_accessions(reverse_context)
            elif variant == "wrong_protein":
                reverse_context = wrong_context
            reverse = _reason_with_context_text(
                var_a=var_b,
                var_b=var_a,
                entity_a=b_ref,
                entity_b=a_ref,
                context_text=reverse_context,
                context_size=len(reverse_records),
                vocabulary=vocabulary,
            )
            agg = aggregate_passes(var_a, var_b, forward, reverse)
            claims.append(agg.as_dict())
    return claims


def _quality(claims: list[dict[str, Any]]) -> dict[str, Any]:
    _data, true_graph = load_sachs_dataset()
    true_edges = {tuple(edge) for edge in true_graph.edges()}
    records = [
        ClaimRecord(
            var_a=str(claim["var_a"]),
            var_b=str(claim["var_b"]),
            cause=str(claim["cause"]),
            effect=str(claim["effect"]),
            confidence=float(claim["confidence"]),
            constraint_type=str(claim["constraint_type"]),
            source="adversarial",
        )
        for claim in claims
    ]
    return compute_quality(
        records,
        true_edges,
        confidence_threshold=0.7,
        n_total_pairs=true_graph.number_of_nodes() * (true_graph.number_of_nodes() - 1),
    )


def main() -> None:
    client = ReactomeClient()
    variants = ("shuffled_accession", "wrong_protein")
    out: dict[str, Any] = {
        "dataset": "sachs",
        "cache_dir": CACHE_DIR.as_posix(),
        "variants": {},
    }
    for variant in variants:
        claims = _run_variant(variant, client)
        out["variants"][variant] = {
            "pairs": claims,
            "quality": _quality(claims),
        }
    _atomic_write_json(OUTPUT_PATH, out)
    print(
        json.dumps(
            {k: v["quality"] for k, v in out["variants"].items()},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
