"""Regression tests for the LLM reasoning path (Step 4 Phases 2 + 3)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from reactome.client import EntityRef, ReactionRecord
from reasoning.reason import (
    CONFLICT_PENALTY,
    CausalClaim,
    aggregate_passes,
    build_reason_messages,
    parse_claim_response,
    reason_pair,
)


def _record(reaction_id: str = "R-HSA-1") -> ReactionRecord:
    return ReactionRecord(
        reaction_id=reaction_id,
        reaction_name=f"reaction {reaction_id}",
        pathway="MAPK",
        role_a="catalyst",
        role_b="output",
        reaction_type="BiochemicalReaction",
        sign=None,
    )


def _llm_response(content: str, model: str = "test/model") -> dict[str, object]:
    return {
        "id": "fixture",
        "model": model,
        "choices": [{"message": {"content": content, "role": "assistant"}}],
    }


VOCAB_3 = ["praf", "pmek", "p44/42"]


class BuildReasonMessagesTests(unittest.TestCase):
    def test_user_message_includes_vocabulary_and_context(self) -> None:
        msgs = build_reason_messages(
            name_a="praf",
            name_b="pmek",
            ids_a=["P04049"],
            ids_b=["Q02750", "P36507"],
            formatted_context="In MAPK pathway, praf catalyses ...",
            vocabulary=VOCAB_3,
        )
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["role"], "user")
        self.assertIn("praf", msgs[1]["content"])
        self.assertIn("Q02750", msgs[1]["content"])
        self.assertIn("In MAPK pathway", msgs[1]["content"])
        for col in VOCAB_3:
            self.assertIn(col, msgs[1]["content"])
        self.assertIn("ANY family isoform counts", msgs[1]["content"])
        self.assertIn("inputs to the same reaction", msgs[1]["content"])


class ParseClaimResponseTests(unittest.TestCase):
    def test_valid_response_emits_constraint_per_confidence_rule(self) -> None:
        content = json.dumps(
            {
                "cause": "praf",
                "effect": "pmek",
                "confidence": 0.92,
                "constraint_type": "hard_required",
                "supporting_reactions": ["R-HSA-1"],
                "contradicting_reactions": [],
                "reasoning": "Direct phosphorylation cascade.",
            }
        )
        claim = parse_claim_response(
            content,
            var_a="praf",
            var_b="pmek",
            vocabulary=VOCAB_3,
            context_size=3,
            served_model="test/model",
            supporting_pool={"R-HSA-1"},
        )
        self.assertEqual(claim.cause, "praf")
        self.assertEqual(claim.effect, "pmek")
        self.assertEqual(claim.constraint_type, "hard_required")
        self.assertEqual(claim.supporting_reactions, ["R-HSA-1"])
        self.assertEqual(claim.notes, [])

    def test_vocab_violation_rewrites_to_unknown_and_logs_note(self) -> None:
        content = json.dumps(
            {
                "cause": "RAF1",
                "effect": "MEK1",
                "confidence": 0.95,
                "constraint_type": "hard_required",
                "supporting_reactions": ["R-HSA-1"],
                "contradicting_reactions": [],
                "reasoning": "x",
            }
        )
        claim = parse_claim_response(
            content,
            var_a="praf",
            var_b="pmek",
            vocabulary=VOCAB_3,
            context_size=2,
            served_model=None,
            supporting_pool={"R-HSA-1"},
        )
        self.assertEqual(claim.cause, "unknown")
        self.assertEqual(claim.effect, "unknown")
        self.assertEqual(claim.constraint_type, "unknown")
        self.assertEqual(claim.raw_llm_cause, "RAF1")
        self.assertEqual(claim.raw_llm_effect, "MEK1")
        self.assertTrue(any(n.startswith("vocab_violation") for n in claim.notes))

    def test_supporting_reactions_pruned_to_context_pool(self) -> None:
        content = json.dumps(
            {
                "cause": "praf",
                "effect": "pmek",
                "confidence": 0.8,
                "constraint_type": "soft_prior",
                "supporting_reactions": ["R-HSA-1", "R-HSA-HALLUCINATED"],
                "contradicting_reactions": ["R-HSA-OOC"],
                "reasoning": "x",
            }
        )
        claim = parse_claim_response(
            content,
            var_a="praf",
            var_b="pmek",
            vocabulary=VOCAB_3,
            context_size=1,
            served_model=None,
            supporting_pool={"R-HSA-1"},
        )
        self.assertEqual(claim.supporting_reactions, ["R-HSA-1"])
        self.assertEqual(claim.contradicting_reactions, [])
        self.assertTrue(any(n.startswith("support_pruned") for n in claim.notes))
        self.assertTrue(any(n.startswith("contradict_pruned") for n in claim.notes))

    def test_unknown_direction_clamps_confidence(self) -> None:
        content = json.dumps(
            {
                "cause": "unknown",
                "effect": "unknown",
                "confidence": 0.85,
                "constraint_type": "soft_prior",
                "supporting_reactions": [],
                "contradicting_reactions": [],
                "reasoning": "ambiguous",
            }
        )
        claim = parse_claim_response(
            content,
            var_a="praf",
            var_b="pmek",
            vocabulary=VOCAB_3,
            context_size=2,
            served_model=None,
            supporting_pool=set(),
        )
        self.assertEqual(claim.cause, "unknown")
        self.assertEqual(claim.constraint_type, "unknown")
        self.assertLess(claim.confidence, 0.5)

    def test_fenced_codeblock_response_parses(self) -> None:
        inner = json.dumps(
            {
                "cause": "praf",
                "effect": "pmek",
                "confidence": 0.7,
                "constraint_type": "soft_prior",
                "supporting_reactions": [],
                "contradicting_reactions": [],
                "reasoning": "x",
            }
        )
        fenced = "```json\n" + inner + "\n```"
        claim = parse_claim_response(
            fenced,
            var_a="praf",
            var_b="pmek",
            vocabulary=VOCAB_3,
            context_size=1,
            served_model=None,
            supporting_pool=set(),
        )
        self.assertEqual(claim.constraint_type, "soft_prior")


class ReasonPairTests(unittest.TestCase):
    def test_empty_context_skips_llm_and_returns_no_context_claim(self) -> None:
        called = {"count": 0}

        def fake_llm(
            _messages: list[dict[str, object]], _cache: Path
        ) -> dict[str, object]:
            called["count"] += 1
            return _llm_response("{}")

        a = EntityRef(kind="protein", ids=["P17252"], display_name="PKC")
        b = EntityRef(kind="protein", ids=["P04049"], display_name="praf")
        claim = reason_pair(
            var_a="PKC",
            var_b="praf",
            entity_a=a,
            entity_b=b,
            reactome_context=[],
            vocabulary=VOCAB_3,
            llm_fetch=fake_llm,
        )
        self.assertEqual(claim.constraint_type, "no_context")
        self.assertEqual(claim.cause, "unknown")
        self.assertEqual(
            called["count"],
            0,
            "LLM must NOT be called when reactome_context is empty.",
        )

    def test_non_empty_context_calls_llm_and_returns_parsed_claim(self) -> None:
        captured: list[list[dict[str, object]]] = []

        def fake_llm(
            messages: list[dict[str, object]], _cache: Path
        ) -> dict[str, object]:
            captured.append(messages)
            return _llm_response(
                json.dumps(
                    {
                        "cause": "praf",
                        "effect": "pmek",
                        "confidence": 0.91,
                        "constraint_type": "hard_required",
                        "supporting_reactions": ["R-HSA-1"],
                        "contradicting_reactions": [],
                        "reasoning": "RAF→MEK direct.",
                    }
                ),
                model="deepseek-ai/DeepSeek-V4-Pro",
            )

        a = EntityRef(kind="protein", ids=["P04049"], display_name="praf")
        b = EntityRef(kind="protein", ids=["Q02750", "P36507"], display_name="pmek")
        claim = reason_pair(
            var_a="praf",
            var_b="pmek",
            entity_a=a,
            entity_b=b,
            reactome_context=[_record("R-HSA-1")],
            vocabulary=VOCAB_3,
            llm_fetch=fake_llm,
        )
        self.assertEqual(len(captured), 1)
        self.assertEqual(claim.cause, "praf")
        self.assertEqual(claim.effect, "pmek")
        self.assertEqual(claim.constraint_type, "hard_required")
        self.assertEqual(claim.supporting_reactions, ["R-HSA-1"])
        self.assertEqual(claim.served_model, "deepseek-ai/DeepSeek-V4-Pro")
        self.assertEqual(claim.reactome_context_size, 1)


def _claim(
    var_a: str,
    var_b: str,
    cause: str,
    effect: str,
    confidence: float,
    *,
    supporting: list[str] | None = None,
) -> CausalClaim:
    return CausalClaim(
        var_a=var_a,
        var_b=var_b,
        cause=cause,
        effect=effect,
        confidence=confidence,
        constraint_type="hard_required" if confidence >= 0.9 else "soft_prior",
        supporting_reactions=supporting or [],
        contradicting_reactions=[],
        reasoning="x",
        reactome_context_size=2,
        served_model="test/model",
    )


class AggregatePassesTests(unittest.TestCase):
    def test_matching_directions_take_higher_confidence(self) -> None:
        forward = _claim("praf", "pmek", "praf", "pmek", 0.80, supporting=["R-HSA-1"])
        reverse = _claim("pmek", "praf", "praf", "pmek", 0.92, supporting=["R-HSA-2"])
        agg = aggregate_passes("praf", "pmek", forward, reverse)
        self.assertEqual(agg.cause, "praf")
        self.assertEqual(agg.effect, "pmek")
        self.assertEqual(agg.confidence, 0.92)
        self.assertEqual(agg.constraint_type, "hard_required")
        self.assertIsNone(agg.conflict)
        self.assertEqual(agg.supporting_reactions, ["R-HSA-1", "R-HSA-2"])

    def test_conflicting_directions_apply_conflict_penalty(self) -> None:
        forward = _claim("praf", "pmek", "praf", "pmek", 0.80)
        reverse = _claim("pmek", "praf", "pmek", "praf", 0.70)
        agg = aggregate_passes("praf", "pmek", forward, reverse)
        self.assertEqual(agg.cause, "praf")
        self.assertEqual(agg.effect, "pmek")
        expected = round(min(0.80, 0.70) * CONFLICT_PENALTY, 4)
        self.assertEqual(agg.confidence, expected)
        self.assertIsNotNone(agg.conflict)
        self.assertEqual(agg.conflict["winner"]["cause"], "praf")  # type: ignore[index]

    def test_conflict_lower_confidence_winner(self) -> None:
        forward = _claim("praf", "pmek", "praf", "pmek", 0.5)
        reverse = _claim("pmek", "praf", "pmek", "praf", 0.95)
        agg = aggregate_passes("praf", "pmek", forward, reverse)
        self.assertEqual(agg.cause, "pmek")
        self.assertEqual(agg.effect, "praf")
        self.assertIsNotNone(agg.conflict)

    def test_one_pass_no_context_uses_other(self) -> None:
        no_ctx = CausalClaim(
            var_a="praf",
            var_b="pmek",
            cause="unknown",
            effect="unknown",
            confidence=0.0,
            constraint_type="no_context",
            supporting_reactions=[],
            contradicting_reactions=[],
            reasoning="empty",
            reactome_context_size=0,
            served_model=None,
        )
        directional = _claim(
            "pmek", "praf", "praf", "pmek", 0.85, supporting=["R-HSA-3"]
        )
        agg = aggregate_passes("praf", "pmek", no_ctx, directional)
        self.assertEqual(agg.cause, "praf")
        self.assertEqual(agg.effect, "pmek")
        self.assertAlmostEqual(agg.confidence, 0.85)

    def test_both_no_context_returns_no_context_aggregate(self) -> None:
        empty = CausalClaim(
            var_a="x",
            var_b="y",
            cause="unknown",
            effect="unknown",
            confidence=0.0,
            constraint_type="no_context",
            supporting_reactions=[],
            contradicting_reactions=[],
            reasoning="empty",
            reactome_context_size=0,
            served_model=None,
        )
        agg = aggregate_passes("praf", "pmek", empty, empty)
        self.assertEqual(agg.constraint_type, "no_context")
        self.assertEqual(agg.cause, "unknown")
        self.assertEqual(agg.confidence, 0.0)

    def test_unknown_in_both_passes_keeps_unknown(self) -> None:
        forward = CausalClaim(
            var_a="praf",
            var_b="pmek",
            cause="unknown",
            effect="unknown",
            confidence=0.40,
            constraint_type="unknown",
            supporting_reactions=[],
            contradicting_reactions=[],
            reasoning="ambiguous",
            reactome_context_size=3,
            served_model="m",
        )
        reverse = CausalClaim(
            var_a="pmek",
            var_b="praf",
            cause="unknown",
            effect="unknown",
            confidence=0.30,
            constraint_type="unknown",
            supporting_reactions=[],
            contradicting_reactions=[],
            reasoning="ambiguous",
            reactome_context_size=3,
            served_model="m",
        )
        agg = aggregate_passes("praf", "pmek", forward, reverse)
        self.assertEqual(agg.cause, "unknown")
        self.assertEqual(agg.constraint_type, "unknown")


if __name__ == "__main__":
    unittest.main()
