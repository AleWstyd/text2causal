"""Regression tests for variable grounding (Step 3)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from grounding.evaluate import evaluate_grounding
from grounding.ground import (
    Grounding as G,
    build_grounding_messages,
    ground_columns,
    parse_grounding_response,
)
from llm.cache import cache_key
from reactome.client import EntityRef, ReactomeClient

FIXTURES = Path(__file__).parent / "fixtures" / "reactome"


def _load_fixture(name: str) -> object:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _seed_cache(
    cache_dir: Path, path: str, params: dict[str, object], payload: object
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key({"path": path, "params": params})
    (cache_dir / f"{key}.json").write_text(json.dumps(payload), encoding="utf-8")


class GroundingPromptTests(unittest.TestCase):
    def test_prompt_assembly_includes_columns_and_hints(self) -> None:
        columns = [
            "PIP2",
            "PIP3",
            "PKA",
            "PKC",
            "P38",
            "p44/42",
            "pakts473",
            "pjnk",
            "plcg",
            "pmek",
            "praf",
        ]
        hints = "Human immune-cell signalling MAPK PI3K PIP phospholipids interventional Sachs"
        messages = build_grounding_messages(
            dataset_description="Sachs 7466 observations 11 phosphorylated modalities",
            domain_hint=hints,
            columns=columns,
        )
        joined = json.dumps(messages)
        self.assertEqual(len(columns), 11)
        for col in columns:
            self.assertIn(col, joined)
        self.assertIn(hints, joined)
        self.assertIn("7466", joined)


class GroundingParserTests(unittest.TestCase):
    def test_response_parser_handles_fenced_codeblock(self) -> None:
        inner = json.dumps(
            {
                "praf": {
                    "kind": "protein",
                    "ids": ["P04049"],
                    "canonical_name": "RAF1",
                    "gene_names": ["RAF1"],
                    "confidence": 0.9,
                    "reasoning": "Phospho-RAF1 maps to RAF1.",
                },
            },
        )
        fenced = "```json\n" + inner + "\n```"
        parsed = parse_grounding_response(fenced)
        self.assertEqual(parsed["praf"]["ids"], ["P04049"])

    def test_response_parser_rejects_malformed(self) -> None:
        with self.assertRaises(ValueError):
            parse_grounding_response("not json")


class GroundingValidationTests(unittest.TestCase):
    def _openai_fixture(self, content: str) -> dict[str, object]:
        return {
            "id": "fixture",
            "model": "test/model-under-free-router",
            "choices": [{"message": {"content": content}}],
        }

    def test_validation_unvalidated_caps_confidence(self) -> None:
        inner = json.dumps(
            {
                "c1": {
                    "kind": "protein",
                    "ids": ["P04049"],
                    "canonical_name": "RAF1",
                    "gene_names": ["RAF1"],
                    "confidence": 0.95,
                    "reasoning": "x",
                },
            },
        )
        mock_rc = MagicMock(spec=ReactomeClient)
        mock_rc.get_entity_info.return_value = None

        preds = ground_columns(
            ["c1"],
            "dataset",
            "domain",
            llm_fetch=lambda _m, _c: self._openai_fixture(inner),
            reactome_client=mock_rc,
            cache_dir=Path("/dev/null/nonexistent-unused"),
        )
        g = preds["c1"]
        self.assertFalse(g.reactome_validated)
        self.assertLessEqual(g.confidence, 0.4)

    def test_validation_validated_preserves_confidence(self) -> None:
        inner = json.dumps(
            {
                "c1": {
                    "kind": "protein",
                    "ids": ["P04049"],
                    "canonical_name": "RAF1",
                    "gene_names": [],
                    "confidence": 0.93,
                    "reasoning": "x",
                },
            },
        )
        mock_rc = MagicMock(spec=ReactomeClient)
        mock_rc.get_entity_info.return_value = {
            "reaction_count": 3,
            "first_reaction_name": "rxn",
        }

        preds = ground_columns(
            ["c1"],
            "dataset",
            "domain",
            llm_fetch=lambda _m, _c: self._openai_fixture(inner),
            reactome_client=mock_rc,
        )
        self.assertAlmostEqual(preds["c1"].confidence, 0.93)


class EvaluateGroundingTests(unittest.TestCase):
    def test_evaluate_grounding_jaccard_and_recall(self) -> None:
        pred = G(
            column="pkc",
            kind="family",
            ids=["P17252", "P05771"],
            canonical_name="PKC subset",
            gene_names=[],
            confidence=1.0,
            reasoning="",
            reactome_validated=True,
            served_model=None,
        )
        gold = {
            "pkc": {
                "kind": "family",
                "ids": ["P17252", "P05771", "P05129"],
                "canonical_name": "PKC",
            },
        }
        metrics = evaluate_grounding({"pkc": pred}, gold)
        self.assertAlmostEqual(metrics["column_recall"], 2 / 3)
        self.assertAlmostEqual(metrics["column_precision"], 1.0)
        self.assertAlmostEqual(metrics["mean_jaccard"], 2 / 3)

    def test_evaluate_grounding_kind_mismatch_logged(self) -> None:
        pred = G(
            column="praf",
            kind="family",
            ids=["P04049"],
            canonical_name="RAF1 family tag",
            gene_names=["RAF1"],
            confidence=1.0,
            reasoning="",
            reactome_validated=True,
            served_model=None,
        )
        gold = {
            "praf": {
                "kind": "protein",
                "ids": ["P04049"],
                "canonical_name": "RAF1",
            },
        }
        metrics = evaluate_grounding({"praf": pred}, gold)
        mismatches = [f for f in metrics["failures"] if f["reason"] == "kind_mismatch"]
        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0]["column"], "praf")

    def test_phosphoinositide_dual_form_required(self) -> None:
        pred = G(
            column="PIP2",
            kind="metabolite",
            ids=["CHEBI:18348"],
            canonical_name="PIP2",
            gene_names=[],
            confidence=0.9,
            reasoning="",
            reactome_validated=True,
            served_model=None,
        )
        gold = {
            "PIP2": {
                "kind": "metabolite",
                "ids": ["CHEBI:18348", "CHEBI:58456"],
                "canonical_name": "PIP2",
            },
        }
        metrics = evaluate_grounding({"PIP2": pred}, gold)
        self.assertAlmostEqual(metrics["column_recall"], 0.5)
        pis = [f for f in metrics["failures"] if f["reason"] == "missing_chebi_form"]
        self.assertEqual(len(pis), 1)
        self.assertEqual(pis[0]["column"], "PIP2")


class GetEntityInfoTests(unittest.TestCase):
    def test_get_entity_info_protein_returns_count(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            _seed_cache(
                cache_dir,
                "/data/mapping/UniProt/P04049/reactions",
                {},
                _load_fixture("mapping_uniprot_P04049.json"),
            )
            client = ReactomeClient(cache_dir=cache_dir)
            with patch("reactome.client.httpx.get") as mocked_get:
                info = client.get_entity_info(EntityRef("protein", ["P04049"], "RAF1"))
                mocked_get.assert_not_called()
            self.assertIsNotNone(info)
            assert info is not None  # narrowing for mypy / type checkers
            self.assertGreaterEqual(info["reaction_count"], 1)
            self.assertIn("first_reaction_name", info)

    def test_get_entity_info_unknown_protein_returns_none(self) -> None:
        client = ReactomeClient()
        with patch.object(ReactomeClient, "reactions_for_protein", return_value=[]):
            ref = EntityRef("protein", ["NOTREAL999"], "?")
            self.assertIsNone(client.get_entity_info(ref))

    def test_get_entity_info_early_exits_after_first_validating_id(self) -> None:
        from unittest.mock import call

        client = ReactomeClient()
        with patch.object(
            ReactomeClient,
            "reactions_for_protein",
            return_value=[
                # one fixture-shaped meta is enough to mark merged non-empty
                __import__("reactome.client", fromlist=["ReactionMeta"]).ReactionMeta(
                    st_id="R-HSA-1",
                    display_name="dummy",
                    species="Homo sapiens",
                ),
            ],
        ) as mock_rfp:
            ref = EntityRef("protein", ["P_FIRST", "P_SECOND", "P_THIRD"], "fam")
            info = client.get_entity_info(ref)
            self.assertIsNotNone(info)
            self.assertEqual(mock_rfp.call_args_list, [call("P_FIRST")])

    def test_get_entity_info_swallows_request_error_per_accession(self) -> None:
        import httpx

        from reactome.client import ReactionMeta

        ok_meta = ReactionMeta(
            st_id="R-HSA-OK", display_name="ok-rxn", species="Homo sapiens"
        )
        client = ReactomeClient()

        def side_effect(accession: str) -> list[ReactionMeta]:
            if accession == "P_BAD":
                raise httpx.ConnectTimeout("simulated")
            return [ok_meta]

        with patch.object(
            ReactomeClient, "reactions_for_protein", side_effect=side_effect
        ):
            ref = EntityRef("protein", ["P_BAD", "P_GOOD"], "fam")
            info = client.get_entity_info(ref)
            self.assertIsNotNone(info)
            assert info is not None
            self.assertEqual(info["reaction_count"], 1)
            self.assertEqual(info["first_reaction_name"], "ok-rxn")


if __name__ == "__main__":
    unittest.main()
