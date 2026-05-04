"""Tests for Step 5 Phase 1 — :class:`~constraints.constraint_builder.ConstraintBuilder`."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from causallearn.graph.GraphNode import GraphNode

from constraints.constraint_builder import (
    ClaimRecord,
    ConstraintBuilder,
    build_lingam_prior_knowledge,
    build_pc_background_knowledge,
    load_priors,
)

_FIX = Path(__file__).resolve().parent / "fixtures" / "builder"


class LoadPriorsTests(unittest.TestCase):
    def test_load_priors_roundtrip_one_of_each_constraint_type(self) -> None:
        recs = load_priors(_FIX / "llm_five_types.json")
        self.assertEqual(len(recs), 5)
        types = {r.constraint_type for r in recs}
        self.assertEqual(
            types,
            {
                "hard_required",
                "soft_prior",
                "hard_forbidden_reverse",
                "unknown",
                "no_context",
            },
        )
        hr = next(r for r in recs if r.constraint_type == "hard_required")
        self.assertEqual(
            (hr.var_a, hr.var_b, hr.cause, hr.effect), ("A", "B", "A", "B")
        )
        self.assertAlmostEqual(hr.confidence, 0.9)
        self.assertEqual(hr.source, "reactome_llm")

    def test_source_reactome_llm_from_top_level_served_models(self) -> None:
        self.assertEqual(
            load_priors(_FIX / "llm_five_types.json")[0].source, "reactome_llm"
        )

    def test_source_reactome_llm_from_claim_level_served_models(self) -> None:
        self.assertEqual(
            load_priors(_FIX / "claims_array_one.json")[0].source, "reactome_llm"
        )

    def test_source_omnipath_all_when_no_served_models(self) -> None:
        self.assertEqual(
            load_priors(_FIX / "omnipath_one.json")[0].source, "omnipath_all"
        )

    def test_source_omnipath_reactome_only_from_filename(self) -> None:
        self.assertEqual(
            load_priors(_FIX / "omnipath_reactome_only_min.json")[0].source,
            "omnipath_reactome_only",
        )

    def test_per_record_source_oracle(self) -> None:
        payload = {
            "pairs": [
                {
                    "var_a": "A",
                    "var_b": "B",
                    "cause": "A",
                    "effect": "B",
                    "confidence": 1.0,
                    "constraint_type": "hard_required",
                    "source": "oracle",
                }
            ]
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "priors.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            recs = load_priors(path)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].source, "oracle")

    def test_per_record_source_freetext_llm(self) -> None:
        payload = {
            "pairs": [
                {
                    "var_a": "X",
                    "var_b": "Y",
                    "cause": "X",
                    "effect": "Y",
                    "confidence": 0.5,
                    "constraint_type": "soft_prior",
                    "source": "freetext_llm",
                }
            ]
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "priors.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            recs = load_priors(path)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].source, "freetext_llm")

    def test_per_record_source_overrides_file_detection(self) -> None:
        """Filename and served_models would imply reactome_llm; record source wins."""
        payload = {
            "background_text_path": "data/sachs/background.txt",
            "served_models": ["deepseek-ai/DeepSeek-V4-Pro"],
            "pairs": [
                {
                    "var_a": "A",
                    "var_b": "B",
                    "cause": "A",
                    "effect": "B",
                    "confidence": 1.0,
                    "constraint_type": "hard_required",
                    "source": "oracle",
                    "served_models": ["deepseek-ai/DeepSeek-V4-Pro"],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "freetext_priors_sachs.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            recs = load_priors(path)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].source, "oracle")


class SummaryAccountingTests(unittest.TestCase):
    def test_summary_invariants_across_thresholds(self) -> None:
        priors = load_priors(_FIX / "llm_five_types.json")
        variables = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        for thr in (0.0, 0.5, 0.7, 0.9, 1.0):
            with self.subTest(threshold=thr):
                b = ConstraintBuilder(priors, variables, thr)
                s = b.summary()
                self.assertEqual(s["total"], len(priors))
                tsum = (
                    s["hard_required"]
                    + s["soft_prior"]
                    + s["hard_forbidden_reverse"]
                    + s["unknown"]
                    + s["no_context"]
                )
                self.assertEqual(tsum, s["total"])
                fsum = (
                    s["kept_required"]
                    + s["kept_forbidden"]
                    + s["discarded_below_threshold"]
                    + s["discarded_unknown"]
                )
                self.assertEqual(fsum, s["total"])

    def test_confidence_equal_to_threshold_is_kept(self) -> None:
        claim = ClaimRecord(
            var_a="M",
            var_b="N",
            cause="M",
            effect="N",
            confidence=0.7,
            constraint_type="soft_prior",
            source="omnipath_all",
        )
        b = ConstraintBuilder([claim], ["M", "N"], 0.7)
        req, forb = b.filtered_priors()
        self.assertEqual(req, [("M", "N")])
        self.assertEqual(forb, [])
        s = b.summary()
        self.assertEqual(s["discarded_below_threshold"], 0)
        self.assertEqual(s["kept_required"], 1)


class FilteredPriorsTests(unittest.TestCase):
    def test_hard_forbidden_reverse_forbids_reverse_only(self) -> None:
        claim = ClaimRecord(
            var_a="E",
            var_b="F",
            cause="E",
            effect="F",
            confidence=1.0,
            constraint_type="hard_forbidden_reverse",
            source="reactome_llm",
        )
        b = ConstraintBuilder([claim], ["E", "F"], 0.5)
        req, forb = b.filtered_priors()
        self.assertEqual(req, [])
        self.assertEqual(forb, [("F", "E")])

    def test_soft_and_hard_required_at_threshold_point_six(self) -> None:
        priors = [
            ClaimRecord(
                var_a="A",
                var_b="B",
                cause="A",
                effect="B",
                confidence=0.65,
                constraint_type="hard_required",
                source="x",
            ),
            ClaimRecord(
                var_a="C",
                var_b="D",
                cause="C",
                effect="D",
                confidence=0.65,
                constraint_type="soft_prior",
                source="x",
            ),
        ]
        b = ConstraintBuilder(priors, ["A", "B", "C", "D"], 0.6)
        req, _ = b.filtered_priors()
        self.assertCountEqual(req, [("A", "B"), ("C", "D")])

    def test_same_ordered_edge_required_and_forbidden_raises(self) -> None:
        priors = [
            ClaimRecord(
                var_a="A",
                var_b="B",
                cause="A",
                effect="B",
                confidence=1.0,
                constraint_type="hard_required",
                source="x",
            ),
            ClaimRecord(
                var_a="A",
                var_b="B",
                cause="B",
                effect="A",
                confidence=1.0,
                constraint_type="hard_forbidden_reverse",
                source="x",
            ),
        ]
        # hard_forbidden_reverse B,A => forbid (A,B); conflicts with required (A,B)
        b = ConstraintBuilder(priors, ["A", "B"], 0.5)
        with self.assertRaisesRegex(
            ValueError, "both required and forbidden|Same ordered edge"
        ):
            b.filtered_priors()

    def test_required_and_forbidden_reverse_nonconflicting_is_ok(self) -> None:
        priors = [
            ClaimRecord(
                var_a="A",
                var_b="B",
                cause="A",
                effect="B",
                confidence=1.0,
                constraint_type="hard_required",
                source="x",
            ),
            ClaimRecord(
                var_a="A",
                var_b="B",
                cause="A",
                effect="B",
                confidence=1.0,
                constraint_type="hard_forbidden_reverse",
                source="x",
            ),
        ]
        b = ConstraintBuilder(priors, ["A", "B"], 0.5)
        req, forb = b.filtered_priors()
        self.assertEqual(req, [("A", "B")])
        self.assertEqual(forb, [("B", "A")])
        b.to_prior_knowledge()


class ToPriorKnowledgeTests(unittest.TestCase):
    def test_opposing_required_edges_raises(self) -> None:
        priors = [
            ClaimRecord(
                var_a="A",
                var_b="B",
                cause="A",
                effect="B",
                confidence=1.0,
                constraint_type="hard_required",
                source="x",
            ),
            ClaimRecord(
                var_a="A",
                var_b="B",
                cause="B",
                effect="A",
                confidence=1.0,
                constraint_type="hard_required",
                source="x",
            ),
        ]
        b = ConstraintBuilder(priors, ["A", "B"], 0.5)
        with self.assertRaisesRegex(ValueError, "opposing required edges"):
            b.to_prior_knowledge()

    def test_unknown_variable_delegates_to_validator(self) -> None:
        claim = ClaimRecord(
            var_a="A",
            var_b="B",
            cause="A",
            effect="B",
            confidence=1.0,
            constraint_type="hard_required",
            source="x",
        )
        b = ConstraintBuilder([claim], ["A"], 0.5)
        with self.assertRaisesRegex(ValueError, "unknown variables"):
            b.to_prior_knowledge()


class UnknownAndWeirdTests(unittest.TestCase):
    def test_discarded_unknown_never_kept(self) -> None:
        variables = ["P", "Q", "R", "S"]
        cases = (
            ("unknown", "unknown", "unknown", "unknown"),
            ("no_context", "P", "Q", "I"),
            ("hard_required", "unknown", "Q", "R"),
        )
        for ctype, cause, effect, va in cases:
            with self.subTest(ctype=ctype):
                claim = ClaimRecord(
                    var_a=va,
                    var_b="S",
                    cause=cause,
                    effect=effect,
                    confidence=1.0,
                    constraint_type=ctype,
                    source="x",
                )
                b = ConstraintBuilder([claim], variables, 0.0)
                s = b.summary()
                self.assertEqual(s["kept_required"], 0)
                self.assertEqual(s["kept_forbidden"], 0)
                self.assertEqual(s["discarded_unknown"], 1)

    def test_unrecognised_constraint_type_raises(self) -> None:
        claim = ClaimRecord(
            var_a="A",
            var_b="B",
            cause="A",
            effect="B",
            confidence=1.0,
            constraint_type="weird",
            source="x",
        )
        b = ConstraintBuilder([claim], ["A", "B"], 0.5)
        with self.assertRaisesRegex(ValueError, "Unrecognised constraint_type"):
            b.summary()


class ClassVersusFreeFunctionTests(unittest.TestCase):
    def test_pc_and_lingam_match_free_functions(self) -> None:
        priors = load_priors(_FIX / "llm_five_types.json")
        variables = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        b = ConstraintBuilder(priors, variables, 0.7)
        pk = b.to_prior_knowledge()
        bk_class = b.build_pc_background_knowledge()
        bk_free = build_pc_background_knowledge(pk)
        mat_class = b.build_lingam_prior_matrix()
        mat_free = build_lingam_prior_knowledge(pk, variables)
        self.assertTrue(np.array_equal(mat_class, mat_free))

        for cause, effect in pk.required_edges:
            self.assertTrue(
                bk_class.is_required(GraphNode(cause), GraphNode(effect)),
                msg=f"{cause}->{effect}",
            )
            self.assertTrue(bk_free.is_required(GraphNode(cause), GraphNode(effect)))
        for cause, effect in pk.forbidden_edges:
            self.assertTrue(bk_class.is_forbidden(GraphNode(cause), GraphNode(effect)))
            self.assertTrue(bk_free.is_forbidden(GraphNode(cause), GraphNode(effect)))

    def test_build_ges_post_hoc_edits_matches_filtered_priors(self) -> None:
        priors = load_priors(_FIX / "llm_five_types.json")
        variables = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        b = ConstraintBuilder(priors, variables, 0.7)
        a, f = b.build_ges_post_hoc_edits()
        a2, f2 = b.filtered_priors()
        self.assertEqual(a, a2)
        self.assertEqual(f, f2)


if __name__ == "__main__":
    unittest.main()
