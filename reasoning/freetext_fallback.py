"""Per-pair parametric-knowledge LLM fallback for Reactome no-context pairs (PR2b).

Each unordered no-context pair is queried once; results match the free-text priors
JSON schema and merge via :func:`reasoning.merge_freetext_fallback.merge_with_freetext_fallback`.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from llm.cache import cached_call
from llm.client import MODEL, _get_client
from reasoning._constraint_rule import decide_constraint_type
from reasoning.reason import REASONING_MAX_TOKENS, _unwrap_json

PerPairLLMFetch = Callable[[list[dict[str, Any]], Path], dict[str, Any]]

FREETEXT_PAIR_FALLBACK_SYSTEM = """You are a causal-reasoning expert in molecular cell biology.
You answer from established mechanistic knowledge (no external database text).

CONFIDENCE SCALE
- 0.9–1.0 — well-established mechanism.
- 0.7–0.89 — likely based on canonical pathways.
- 0.5–0.69 — plausible but uncertain.
- Below 0.5 — treat direction as unknown (use constraint_type "unknown" or set cause/effect to "unknown").

Output ONLY valid JSON. No preamble."""


def _freetext_pair_user_message(
    *,
    dataset_description: str,
    vocabulary: list[str],
    var_a: str,
    var_b: str,
) -> str:
    voc = json.dumps(sorted(vocabulary))
    return (
        f"Dataset: {dataset_description}\n"
        f"Allowed variable tokens (use exactly one, or 'unknown'): {voc}\n"
        f"Pair: {var_a} vs {var_b}\n\n"
        f"Based on established molecular cell biology, what is the most likely "
        f"causal direction between {var_a} and {var_b}? Return JSON with cause, "
        f"effect, confidence (float 0..1), constraint_type "
        f"('hard_required'|'soft_prior'|'hard_forbidden_reverse'|'unknown'), "
        f"reasoning (1-2 sentences). Output JSON only."
    )


def _dedupe_unordered_pairs(
    pairs: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for raw_a, raw_b in pairs:
        sa, sb = sorted((str(raw_a), str(raw_b)))
        key = (sa, sb)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return sorted(out, key=lambda t: (t[0], t[1]))


def _response_content(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    msg = first.get("message") or {}
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    return content.strip() if isinstance(content, str) else ""


def _served_model_from_response(response: dict[str, Any]) -> str | None:
    served = response.get("model")
    return str(served) if isinstance(served, str) and served else None


def _default_llm_fetch(
    messages: list[dict[str, Any]],
    cache_dir: Path,
) -> dict[str, Any]:
    """Cached completion; ``task`` discriminates cache keys from Step 4 / other chat calls."""

    payload = {
        "task": "per_pair_freetext_fallback",
        "model": MODEL,
        "messages": messages,
        "max_tokens": REASONING_MAX_TOKENS,
    }

    def fetch() -> dict[str, Any]:
        response = _get_client().chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=REASONING_MAX_TOKENS,
        )
        return response.model_dump()

    return cached_call(cache_dir, payload, fetch)


def generate_per_pair_freetext_priors(
    dataset_name: str,
    dataset_description: str,
    no_context_pairs: list[tuple[str, str]],
    vocabulary: list[str],
    *,
    llm_fetch: PerPairLLMFetch | None = None,
    cache_dir: Path = Path("cache/llm"),
) -> dict[str, Any]:
    """Query the LLM once per unordered no-context pair; return a freetext-priors-shaped dict."""

    vocab_tokens = sorted({str(x) for x in vocabulary})
    vocab_set = set(vocab_tokens) | {"unknown"}
    ordered_pairs = _dedupe_unordered_pairs(no_context_pairs)
    fetch = llm_fetch or _default_llm_fetch

    pairs_out: list[dict[str, Any]] = []
    all_served: list[str] = []
    n_skipped_unknown_var = 0

    type_counts: Counter[str] = Counter()

    for var_a, var_b in ordered_pairs:
        user = _freetext_pair_user_message(
            dataset_description=dataset_description,
            vocabulary=vocab_tokens,
            var_a=var_a,
            var_b=var_b,
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": FREETEXT_PAIR_FALLBACK_SYSTEM},
            {"role": "user", "content": user},
        ]
        response = fetch(messages, cache_dir)
        if not isinstance(response, dict):
            n_skipped_unknown_var += 1
            continue

        served = _served_model_from_response(response)
        if served:
            all_served.append(served)

        content = _response_content(response)
        if not content:
            n_skipped_unknown_var += 1
            continue

        try:
            parsed = json.loads(_unwrap_json(content))
        except (json.JSONDecodeError, ValueError):
            n_skipped_unknown_var += 1
            continue

        if not isinstance(parsed, dict):
            n_skipped_unknown_var += 1
            continue

        raw_cause = str(parsed.get("cause") or "").strip()
        raw_effect = str(parsed.get("effect") or "").strip()
        conf_raw = parsed.get("confidence", 0.0)
        if isinstance(conf_raw, bool):
            conf_raw = 0.0
        confidence = float(conf_raw) if isinstance(conf_raw, (int, float)) else 0.0
        confidence = max(0.0, min(1.0, confidence))

        if raw_cause not in vocab_set or raw_effect not in vocab_set:
            n_skipped_unknown_var += 1
            continue
        if raw_cause == "unknown" or raw_effect == "unknown":
            n_skipped_unknown_var += 1
            continue

        raw_ct = str(parsed.get("constraint_type") or "").strip()
        forbidden = raw_ct == "hard_forbidden_reverse"

        if confidence < 0.5:
            constraint_type: str = "unknown"
        else:
            constraint_type = decide_constraint_type(
                confidence,
                forbidden_reverse=forbidden,
            )

        reasoning_field = parsed.get("reasoning")
        reasoning = reasoning_field.strip() if isinstance(reasoning_field, str) else ""

        served_list = [served] if served else []
        rec: dict[str, Any] = {
            "var_a": var_a,
            "var_b": var_b,
            "cause": raw_cause,
            "effect": raw_effect,
            "confidence": round(confidence, 4),
            "constraint_type": constraint_type,
            "reasoning": reasoning,
            "source": "per_pair_freetext_fallback",
            "served_models": served_list,
        }
        pairs_out.append(rec)
        type_counts[str(constraint_type)] += 1

    served_unique = sorted(set(all_served))
    return {
        "dataset_name": dataset_name,
        "n_pairs_queried": len(ordered_pairs),
        "source_kind": "per_pair_freetext_fallback",
        "n_hard_forbidden_reverse": int(type_counts.get("hard_forbidden_reverse", 0)),
        "n_hard_required": int(type_counts.get("hard_required", 0)),
        "n_relations_emitted": len(pairs_out),
        "n_relations_skipped_unknown_var": n_skipped_unknown_var,
        "n_soft_prior": int(type_counts.get("soft_prior", 0)),
        "n_unknown": int(type_counts.get("unknown", 0)),
        "pairs": pairs_out,
        "served_models": served_unique,
    }
