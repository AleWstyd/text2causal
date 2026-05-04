"""Step 6 Phase 5 — LLM cache cost / token summary (appendix-ready).

Classifies each cached OpenAI-style completion by parsing
``choices[0].message.content`` as JSON. Step 3 grounding does not embed a stage
tag in cache files; we infer the stage from response shape only (Option **b**).

**Classifier rules (verify against `grounding/ground.py` and reasoning prompts):**

1. **Grounding** — Parsed value is either:

   - A dict that directly contains ``kind``, ``canonical_name``, and at least one
     of ``ids`` or ``gene_names`` (all as keys); or
   - A dict whose *values* are all dicts, at least one value exists, and every
     value satisfies the same inner shape (batched Step 3 column → grounding
     object).

2. **Reasoning** — Parsed value is a dict with keys ``cause``, ``effect``,
   ``confidence``, ``constraint_type``, and at least one of
   ``supporting_reactions`` or ``contradicting_reactions``.

3. **Freetext** — Parsed value is a non-empty list; every element is a dict
   containing ``cause``, ``effect``, ``relation_type``, and ``confidence``.

4. **unknown** — Anything else (invalid JSON, wrong root type, fenced but
   unparseable content, etc.).

Optional triple-backtick code fences (including a ``json`` language tag) around the payload are stripped before
``json.loads`` (models often wrap JSON).

**Pricing:** Baseten Model APIs public rate table (snapshot below). The table
lists per-1M-token input and output prices for hosted models including the
DeepSeek V4 family; we use the tier that matches ``deepseek-ai/DeepSeek-V4-Pro``
responses observed in ``cache/llm/``.

**Latency:** Per-call wall time is not stored in cache. We report rough
wall-clock using fixed seconds per classified call (see ``approx_wall_clock_seconds``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

# Per https://www.baseten.co/pricing — Model APIs table (first listed tier:
# input $1.74 / 1M, output $3.48 / 1M) as of page retrieval used for this
# snapshot. DeepSeek V4 / V4-Pro is offered via the same Model API product;
# confirm against Baseten billing if prices change.
PRICE_PER_M_INPUT_USD: float = 1.74
PRICE_PER_M_OUTPUT_USD: float = 3.48
PRICE_SNAPSHOT_DATE: str = "2026-05-04"
PRICE_SOURCE_URL: str = "https://www.baseten.co/pricing"

STAGES: Final[tuple[str, ...]] = ("grounding", "reasoning", "freetext", "unknown")

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

_APPROX_LATENCY_GROUNDING_S: Final[int] = 10
_APPROX_LATENCY_REASONING_S: Final[int] = 30
_APPROX_LATENCY_FREETEXT_S: Final[int] = 30

_LATENCY_DISCLOSURE: Final[str] = (
    "Approximate; per-call latency is not stored in the cache. Multipliers used: "
    "grounding=10s, reasoning=30s, freetext=30s. See experiments/cost_report.py "
    "for details."
)


def _unwrap_json_content(raw: str) -> str:
    """Strip optional ``` / ```json fences; match grounding helper behavior."""
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


def _dict_has_grounding_inner_shape(d: dict[Any, Any]) -> bool:
    if "kind" not in d or "canonical_name" not in d:
        return False
    if "ids" not in d and "gene_names" not in d:
        return False
    return True


def _is_grounding_parsed(parsed: Any) -> bool:
    if not isinstance(parsed, dict) or not parsed:
        return False
    if _dict_has_grounding_inner_shape(parsed):
        return True
    values = list(parsed.values())
    if not all(isinstance(v, dict) for v in values):
        return False
    return all(_dict_has_grounding_inner_shape(v) for v in values)


def _is_reasoning_parsed(parsed: Any) -> bool:
    if not isinstance(parsed, dict):
        return False
    need = {"cause", "effect", "confidence", "constraint_type"}
    if not need.issubset(parsed.keys()):
        return False
    return "supporting_reactions" in parsed or "contradicting_reactions" in parsed


def _is_freetext_list(parsed: Any) -> bool:
    if not isinstance(parsed, list) or not parsed:
        return False
    need_el = {"cause", "effect", "relation_type", "confidence"}
    for el in parsed:
        if not isinstance(el, dict) or not need_el.issubset(el.keys()):
            return False
    return True


def classify_stage(response: dict[str, Any]) -> str:
    """Return ``grounding`` | ``reasoning`` | ``freetext`` | ``unknown``."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return "unknown"
    first = choices[0]
    if not isinstance(first, dict):
        return "unknown"
    msg = first.get("message")
    if not isinstance(msg, dict):
        return "unknown"
    raw_c = msg.get("content")
    if not isinstance(raw_c, str) or not raw_c.strip():
        return "unknown"
    to_parse = _unwrap_json_content(raw_c)
    try:
        parsed = json.loads(to_parse)
    except (json.JSONDecodeError, TypeError, ValueError):
        return "unknown"

    if _is_freetext_list(parsed):
        return "freetext"
    if _is_grounding_parsed(parsed):
        return "grounding"
    if _is_reasoning_parsed(parsed):
        return "reasoning"
    return "unknown"


def _usage_tokens(response: dict[str, Any]) -> tuple[int, int, int]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return 0, 0, 0
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    tt = usage.get("total_tokens", 0)
    try:
        pi, ci, ti = int(pt), int(ct), int(tt)
    except (TypeError, ValueError):
        return 0, 0, 0
    if ti <= 0 and (pi > 0 or ci > 0):
        ti = pi + ci
    return pi, ci, ti


def estimated_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens / 1_000_000.0) * PRICE_PER_M_INPUT_USD + (
        completion_tokens / 1_000_000.0
    ) * PRICE_PER_M_OUTPUT_USD


def walk_cache(cache_dir: Path | None = None) -> list[dict[str, Any]]:
    """One record per ``*.json`` file under ``cache_dir`` (recursive)."""
    base = REPO_ROOT / "cache" / "llm" if cache_dir is None else cache_dir
    if not base.is_dir():
        return []

    paths = sorted(base.rglob("*.json"), key=lambda p: p.as_posix())
    out: list[dict[str, Any]] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        stripped = text.lstrip()
        if stripped and stripped[0] not in "{[":
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue

        stage = classify_stage(payload)
        served = payload.get("model")
        served_model = str(served) if served is not None else ""
        prompt_tokens, completion_tokens, total_tokens = _usage_tokens(payload)
        cost = estimated_cost_usd(prompt_tokens, completion_tokens)
        out.append(
            {
                "path": str(path),
                "stage": stage,
                "served_model": served_model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": cost,
            }
        )
    return out


def _empty_bucket() -> dict[str, Any]:
    return {
        "n_entries": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "estimated_cost_usd": 0.0,
    }


def aggregate(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Nested totals by stage, by served model, and overall."""
    by_stage: dict[str, dict[str, Any]] = {s: _empty_bucket() for s in STAGES}
    by_model: dict[str, dict[str, Any]] = {}
    totals = _empty_bucket()

    for e in entries:
        stage = str(e.get("stage", "unknown"))
        if stage not in by_stage:
            stage = "unknown"
        sm = str(e.get("served_model", ""))
        pt = int(e.get("prompt_tokens", 0))
        ct = int(e.get("completion_tokens", 0))
        cost = float(e.get("estimated_cost_usd", 0.0))

        for bucket in (by_stage[stage], totals):
            bucket["n_entries"] += 1
            bucket["prompt_tokens"] += pt
            bucket["completion_tokens"] += ct
            bucket["estimated_cost_usd"] += cost

        m_bucket = by_model.setdefault(
            sm,
            _empty_bucket(),
        )
        m_bucket["n_entries"] += 1
        m_bucket["prompt_tokens"] += pt
        m_bucket["completion_tokens"] += ct
        m_bucket["estimated_cost_usd"] += cost

    return {
        "by_stage": by_stage,
        "by_served_model": dict(sorted(by_model.items(), key=lambda kv: kv[0])),
        "totals": totals,
    }


def _approx_wall_clock_seconds(by_stage: dict[str, dict[str, Any]]) -> dict[str, int]:
    g = int(by_stage["grounding"]["n_entries"]) * _APPROX_LATENCY_GROUNDING_S
    r = int(by_stage["reasoning"]["n_entries"]) * _APPROX_LATENCY_REASONING_S
    f = int(by_stage["freetext"]["n_entries"]) * _APPROX_LATENCY_FREETEXT_S
    return {
        "grounding": g,
        "reasoning": r,
        "freetext": f,
        "total": g + r + f,
    }


def _build_headline(summary: dict[str, Any]) -> str:
    t = summary["totals"]
    n = int(t["n_entries"])
    pt = int(t["prompt_tokens"])
    ct = int(t["completion_tokens"])
    usd = float(t["estimated_cost_usd"])
    return (
        f"LLM cache: {n} completion(s); {pt:,} prompt + {ct:,} completion tokens; "
        f"estimated ${usd:.4f} USD at Baseten Model API rates "
        f"(${PRICE_PER_M_INPUT_USD}/M in, ${PRICE_PER_M_OUTPUT_USD}/M out, "
        f"snapshot {PRICE_SNAPSHOT_DATE})."
    )


def _round_cost_tree(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _round_cost_tree(v) for k, v in x.items()}
    if isinstance(x, float):
        return round(x, 10)
    return x


def main() -> None:
    cache_dirs = [REPO_ROOT / "cache" / "llm", REPO_ROOT / "cache" / "llm_adversarial"]
    entries: list[dict[str, Any]] = []
    for cache_dir in cache_dirs:
        entries.extend(walk_cache(cache_dir))
    summary = aggregate(entries)
    approx_clock = _approx_wall_clock_seconds(summary["by_stage"])
    headline = _build_headline(summary)

    doc: dict[str, Any] = {
        "snapshot_date": PRICE_SNAPSHOT_DATE,
        "price_per_m_input_usd": PRICE_PER_M_INPUT_USD,
        "price_per_m_output_usd": PRICE_PER_M_OUTPUT_USD,
        "price_source_url": PRICE_SOURCE_URL,
        "cache_dirs": [path.as_posix() for path in cache_dirs if path.is_dir()],
        "summary": summary,
        "approx_wall_clock_seconds": approx_clock,
        "latency_disclosure": _LATENCY_DISCLOSURE,
        "headline": headline,
    }
    doc = _round_cost_tree(doc)

    out_path = REPO_ROOT / "experiments" / "cost_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, indent=2, sort_keys=True) + "\n"
    out_path.write_text(text, encoding="utf-8")
    print(headline)


if __name__ == "__main__":
    main()
