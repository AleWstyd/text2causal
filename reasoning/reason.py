"""LLM causal-direction reasoning over Reactome reaction context (Step 4).

The system prompt is the canonical Step 4 reasoning prompt pinned in
``docs/dev_plan_v2.md`` §8.2. The user message extends it with three small
guard-rails that the smoke pairs proved necessary in practice:

1. The dataset's column-name vocabulary, so the LLM emits ``praf`` not ``RAF1``.
2. An explicit "any isoform counts" reminder for protein families.
3. A reminder that pairs co-appearing only as input do not establish direction.

Per ``docs/step_04_causal_reasoning.md``: when ``reactome_context`` is empty
we *do not* call the LLM — we emit a ``no_context`` claim. Saves tokens and
keeps the priors artefact's ``no_context_pairs`` list trustworthy.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from llm.cache import cached_call
from llm.client import MODEL, _get_client
from reactome.client import EntityRef, ReactionRecord, format_context_for_llm
from reasoning._constraint_rule import ConstraintType, decide_constraint_type

REASONING_SYSTEM = """You are a causal-reasoning expert in molecular cell biology. You are given
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

Output ONLY valid JSON. No preamble."""

REASONING_USER_TEMPLATE = """Entity A: {name_a} ({ids_a})
Entity B: {name_b} ({ids_b})

Reactome reaction context:
{formatted_context}

Allowed names for `cause`/`effect` fields (the dataset's column-name vocabulary;
use exactly one of these tokens or 'unknown'):
{vocabulary}

When the entities are protein families (e.g. PKC, PKA, ERK1/2, JNK, p38, AKT),
co-participation of ANY family isoform counts; do not require all isoforms to
appear. When both entities appear only as inputs to the same reaction, the
direction is NOT establishable — return cause='unknown' / effect='unknown'.

Determine the most likely causal direction. Return:
{{
  "cause": "<one of the allowed names above, or 'unknown'>",
  "effect": "<one of the allowed names above, or 'unknown'>",
  "confidence": <float 0.0-1.0>,
  "constraint_type": "hard_required" | "soft_prior" | "hard_forbidden_reverse" | "unknown",
  "supporting_reactions": ["<R-HSA-* id from the context above>", ...],
  "contradicting_reactions": ["<R-HSA-* id from the context above>", ...],
  "reasoning": "<2-3 sentences>"
}}"""


@dataclass(frozen=True)
class CausalClaim:
    """A single LLM call's directional claim for an ordered (var_a, var_b) pair."""

    var_a: str
    var_b: str
    cause: str
    effect: str
    confidence: float
    constraint_type: ConstraintType
    supporting_reactions: list[str]
    contradicting_reactions: list[str]
    reasoning: str
    reactome_context_size: int
    served_model: str | None = None
    raw_llm_cause: str | None = None
    raw_llm_effect: str | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_reason_messages(
    *,
    name_a: str,
    name_b: str,
    ids_a: list[str],
    ids_b: list[str],
    formatted_context: str,
    vocabulary: list[str],
) -> list[dict[str, str]]:
    """Build the (system, user) message pair for the §8.2 reasoning prompt."""

    user = REASONING_USER_TEMPLATE.format(
        name_a=name_a,
        name_b=name_b,
        ids_a=", ".join(ids_a),
        ids_b=", ".join(ids_b),
        formatted_context=formatted_context,
        vocabulary=json.dumps(sorted(vocabulary)),
    )
    return [
        {"role": "system", "content": REASONING_SYSTEM},
        {"role": "user", "content": user},
    ]


def _unwrap_json(content: str) -> str:
    s = content.strip()
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


def parse_claim_response(
    content: str,
    *,
    var_a: str,
    var_b: str,
    vocabulary: list[str],
    context_size: int,
    served_model: str | None,
    supporting_pool: set[str],
) -> CausalClaim:
    """Parse the LLM JSON, validate against the column-name vocabulary."""

    parsed = json.loads(_unwrap_json(content))
    if not isinstance(parsed, dict):
        raise ValueError("Reasoning JSON root must be an object.")

    raw_cause = str(parsed.get("cause") or "").strip()
    raw_effect = str(parsed.get("effect") or "").strip()
    confidence_raw = parsed.get("confidence", 0.0)
    if isinstance(confidence_raw, bool):
        confidence_raw = 0.0
    confidence = (
        float(confidence_raw) if isinstance(confidence_raw, (int, float)) else 0.0
    )
    confidence = max(0.0, min(1.0, confidence))

    notes: list[str] = []
    vocab_set = set(vocabulary) | {"unknown"}
    cause = raw_cause if raw_cause in vocab_set else "unknown"
    effect = raw_effect if raw_effect in vocab_set else "unknown"
    if raw_cause not in vocab_set or raw_effect not in vocab_set:
        notes.append(
            "vocab_violation: LLM emitted a name outside the allowed vocabulary; "
            "rewritten to 'unknown' per the spec's strict-validation rule."
        )

    if cause == "unknown" or effect == "unknown":
        cause, effect = "unknown", "unknown"
        confidence = min(confidence, 0.49)

    raw_constraint = str(parsed.get("constraint_type") or "").strip()
    forbidden = raw_constraint == "hard_forbidden_reverse"
    constraint = decide_constraint_type(confidence, forbidden_reverse=forbidden)
    if cause == "unknown":
        constraint = "unknown"

    def _string_list(field_name: str) -> list[str]:
        raw = parsed.get(field_name) or []
        if not isinstance(raw, list):
            return []
        return [str(item) for item in raw if isinstance(item, str)]

    supporting = _string_list("supporting_reactions")
    contradicting = _string_list("contradicting_reactions")

    if supporting_pool:
        kept_supporting = [r for r in supporting if r in supporting_pool]
        if len(kept_supporting) != len(supporting):
            notes.append(
                "support_pruned: dropped supporting_reactions not present in the "
                "Reactome context window."
            )
        supporting = kept_supporting
        kept_contradicting = [r for r in contradicting if r in supporting_pool]
        if len(kept_contradicting) != len(contradicting):
            notes.append(
                "contradict_pruned: dropped contradicting_reactions not present "
                "in the Reactome context window."
            )
        contradicting = kept_contradicting

    reasoning_field = parsed.get("reasoning")
    reasoning = reasoning_field.strip() if isinstance(reasoning_field, str) else ""

    return CausalClaim(
        var_a=var_a,
        var_b=var_b,
        cause=cause,
        effect=effect,
        confidence=round(confidence, 4),
        constraint_type=constraint,
        supporting_reactions=supporting,
        contradicting_reactions=contradicting,
        reasoning=reasoning,
        reactome_context_size=context_size,
        served_model=served_model,
        raw_llm_cause=raw_cause or None,
        raw_llm_effect=raw_effect or None,
        notes=notes,
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
        reasoning="No Reactome co-participation evidence; LLM not consulted.",
        reactome_context_size=0,
        served_model=None,
    )


REASONING_MAX_TOKENS = 16384
"""Per-call completion-token budget for the Step 4 LLM.

Deliberately generous: the active model
(``deepseek-ai/DeepSeek-V4-Pro``, see ``llm/client.py:MODEL``) is a chain-of-
thought reasoner whose ``reasoning_content`` field consumes most of the
completion-token budget *before* the actual JSON answer is emitted. The
default endpoint cap of 4096 tokens truncates mid-JSON for these prompts,
so we set ``max_tokens=16384`` to ensure the final JSON always closes.
``REASONING_MAX_TOKENS`` is part of the cache key so that any future
budget bump invalidates only the new entries (existing entries keep
replaying)."""


def default_llm_fetch(
    messages: list[dict[str, Any]],
    *,
    cache_dir: Path,
    model: str = MODEL,
    max_tokens: int = REASONING_MAX_TOKENS,
) -> dict[str, Any]:
    """Cached chat-completion fetch with a Step-4 cache-key discriminator."""

    payload = {
        "task": "step4_reasoning",
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }

    def fetch() -> dict[str, Any]:
        response = _get_client().chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
        )
        return response.model_dump()

    return cached_call(cache_dir, payload, fetch)


LLMFetch = Callable[[list[dict[str, Any]], Path], dict[str, Any]]


def reason_pair(
    *,
    var_a: str,
    var_b: str,
    entity_a: EntityRef,
    entity_b: EntityRef,
    reactome_context: list[ReactionRecord],
    vocabulary: list[str],
    llm_fetch: LLMFetch | None = None,
    cache_dir: Path = Path("cache/llm"),
) -> CausalClaim:
    """Return a single LLM directional claim for the ordered pair (``var_a``, ``var_b``).

    When ``reactome_context`` is empty, no LLM call is made; a ``no_context``
    claim is returned per ``docs/step_04_causal_reasoning.md``.
    """

    if not reactome_context:
        return _no_context_claim(var_a, var_b)

    formatted_context = format_context_for_llm(reactome_context, var_a, var_b)
    messages = build_reason_messages(
        name_a=var_a,
        name_b=var_b,
        ids_a=list(entity_a.ids),
        ids_b=list(entity_b.ids),
        formatted_context=formatted_context,
        vocabulary=vocabulary,
    )

    fetch = llm_fetch or (lambda m, c: default_llm_fetch(m, cache_dir=c))
    response = fetch(messages, cache_dir)

    served = response.get("model") if isinstance(response, dict) else None
    if not isinstance(served, str):
        served = None

    choices = response.get("choices") or []
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"Reasoning LLM response has no choices for {var_a}->{var_b}.")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError(f"Malformed reasoning choice for {var_a}->{var_b}.")
    msg = first.get("message") or {}
    content = msg.get("content") if isinstance(msg, dict) else None
    if not isinstance(content, str):
        raise ValueError(
            f"Reasoning LLM returned non-string content for {var_a}->{var_b}."
        )

    supporting_pool = {r.reaction_id for r in reactome_context if r.reaction_id}
    return parse_claim_response(
        content,
        var_a=var_a,
        var_b=var_b,
        vocabulary=vocabulary,
        context_size=len(reactome_context),
        served_model=served,
        supporting_pool=supporting_pool,
    )


# ----------------------------------------------------------------------
# Aggregation across both A->B and B->A passes
# ----------------------------------------------------------------------


CONFLICT_PENALTY = 0.7


@dataclass(frozen=True)
class AggregatedClaim:
    """Final per-ordered-pair claim after combining both A→B and B→A LLM passes."""

    var_a: str
    var_b: str
    cause: str
    effect: str
    confidence: float
    constraint_type: ConstraintType
    supporting_reactions: list[str]
    contradicting_reactions: list[str]
    reasoning: str
    reactome_context_size: int
    n_passes: int
    served_models: list[str]
    conflict: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _score(claim: CausalClaim, *, want_cause: str, want_effect: str) -> float:
    if claim.cause == want_cause and claim.effect == want_effect:
        return claim.confidence
    return 0.0


def aggregate_passes(
    var_a: str,
    var_b: str,
    forward: CausalClaim,
    reverse: CausalClaim,
) -> AggregatedClaim:
    """Aggregate the (A→B) and (B→A) LLM passes per the doc spec.

    - Same ``cause``/``effect`` → take the higher-confidence call.
    - Conflicting cause/effect with similar confidence → final confidence is
      ``min(c1, c2) * 0.7`` and the conflict is logged.
    """

    served_models = sorted(
        {m for m in (forward.served_model, reverse.served_model) if m}
    )

    if (
        forward.constraint_type == "no_context"
        and reverse.constraint_type == "no_context"
    ):
        return AggregatedClaim(
            var_a=var_a,
            var_b=var_b,
            cause="unknown",
            effect="unknown",
            confidence=0.0,
            constraint_type="no_context",
            supporting_reactions=[],
            contradicting_reactions=[],
            reasoning="No Reactome co-participation evidence on either pass.",
            reactome_context_size=max(
                forward.reactome_context_size, reverse.reactome_context_size
            ),
            n_passes=0,
            served_models=served_models,
            conflict=None,
            notes=[],
        )

    forward_score_ab = _score(forward, want_cause=var_a, want_effect=var_b)
    forward_score_ba = _score(forward, want_cause=var_b, want_effect=var_a)
    reverse_score_ab = _score(reverse, want_cause=var_a, want_effect=var_b)
    reverse_score_ba = _score(reverse, want_cause=var_b, want_effect=var_a)

    score_ab = max(forward_score_ab, reverse_score_ab)
    score_ba = max(forward_score_ba, reverse_score_ba)

    conflict_payload: dict[str, Any] | None = None
    if score_ab > 0 and score_ba > 0:
        winner_dir = (var_a, var_b) if score_ab >= score_ba else (var_b, var_a)
        cause, effect = winner_dir
        c1, c2 = score_ab, score_ba
        confidence = round(min(c1, c2) * CONFLICT_PENALTY, 4)
        conflict_payload = {
            "var_a": var_a,
            "var_b": var_b,
            "forward_pass": {
                "cause": forward.cause,
                "effect": forward.effect,
                "confidence": forward.confidence,
            },
            "reverse_pass": {
                "cause": reverse.cause,
                "effect": reverse.effect,
                "confidence": reverse.confidence,
            },
            "winner": {"cause": cause, "effect": effect},
            "penalty_applied": CONFLICT_PENALTY,
        }
    elif score_ab > 0:
        cause, effect = var_a, var_b
        confidence = score_ab
    elif score_ba > 0:
        cause, effect = var_b, var_a
        confidence = score_ba
    else:
        cause, effect = "unknown", "unknown"
        confidence = round(
            max(
                forward.confidence if forward.cause == "unknown" else 0.0,
                reverse.confidence if reverse.cause == "unknown" else 0.0,
            ),
            4,
        )

    forbidden_reverse = (
        forward.constraint_type == "hard_forbidden_reverse"
        and reverse.constraint_type == "hard_forbidden_reverse"
    )
    constraint = decide_constraint_type(confidence, forbidden_reverse=forbidden_reverse)
    if cause == "unknown":
        constraint = "unknown"

    if cause == var_a and effect == var_b:
        primary = forward if forward_score_ab >= reverse_score_ab else reverse
    elif cause == var_b and effect == var_a:
        primary = forward if forward_score_ba >= reverse_score_ba else reverse
    else:
        primary = forward

    supporting = sorted({*forward.supporting_reactions, *reverse.supporting_reactions})
    contradicting = sorted(
        {*forward.contradicting_reactions, *reverse.contradicting_reactions}
    )

    notes: list[str] = []
    notes.extend(forward.notes)
    notes.extend(reverse.notes)

    return AggregatedClaim(
        var_a=var_a,
        var_b=var_b,
        cause=cause,
        effect=effect,
        confidence=round(confidence, 4),
        constraint_type=constraint,
        supporting_reactions=supporting,
        contradicting_reactions=contradicting,
        reasoning=primary.reasoning,
        reactome_context_size=max(
            forward.reactome_context_size, reverse.reactome_context_size
        ),
        n_passes=int(forward.constraint_type != "no_context")
        + int(reverse.constraint_type != "no_context"),
        served_models=served_models,
        conflict=conflict_payload,
        notes=sorted(set(notes)),
    )
