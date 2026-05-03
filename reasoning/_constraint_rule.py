"""Shared confidence → constraint-type rule for the LLM and OmniPath floor paths.

Pinned in ``docs/dev_plan_v2.md`` §3 Step 4 and ``docs/step_04_causal_reasoning.md``:

- ``confidence >= 0.9`` → ``hard_required`` (or ``hard_forbidden_reverse`` when
  the producer explicitly identified a forbidden direction).
- ``0.6 <= confidence < 0.9`` → ``soft_prior``.
- ``confidence < 0.6`` → ``unknown`` (kept in JSON for analysis but discarded
  for downstream constraint translation).

Both Phase 1 (OmniPath floor) and Phase 3 (LLM aggregation) call this so that
the two priors files use *identical* constraint-type semantics.
"""

from __future__ import annotations

from typing import Literal

ConstraintType = Literal[
    "hard_required",
    "hard_forbidden_reverse",
    "soft_prior",
    "unknown",
    "no_context",
]

HARD_THRESHOLD: float = 0.9
SOFT_THRESHOLD: float = 0.6


def decide_constraint_type(
    confidence: float,
    *,
    forbidden_reverse: bool = False,
) -> ConstraintType:
    """Map a confidence score to one of the canonical constraint-type labels."""

    if confidence is None:
        return "unknown"
    if confidence >= HARD_THRESHOLD:
        return "hard_forbidden_reverse" if forbidden_reverse else "hard_required"
    if confidence >= SOFT_THRESHOLD:
        return "soft_prior"
    return "unknown"
