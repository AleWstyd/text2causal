"""Regression tests for variable grounding (Step 3)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
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
        mock_rc.validate_ids.return_value = []

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
        self.assertEqual(g.ids, [])

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
        mock_rc.validate_ids.return_value = ["P04049"]

        preds = ground_columns(
            ["c1"],
            "dataset",
            "domain",
            llm_fetch=lambda _m, _c: self._openai_fixture(inner),
            reactome_client=mock_rc,
        )
        self.assertAlmostEqual(preds["c1"].confidence, 0.93)
        self.assertEqual(preds["c1"].ids, ["P04049"])

    def test_per_id_drop_keeps_only_validated_accessions(self) -> None:
        inner = json.dumps(
            {
                "fam": {
                    "kind": "family",
                    "ids": ["P04049", "P15056", "Q02750"],
                    "canonical_name": "fam",
                    "gene_names": [],
                    "confidence": 0.9,
                    "reasoning": "x",
                },
            },
        )
        mock_rc = MagicMock(spec=ReactomeClient)
        mock_rc.validate_ids.return_value = ["P04049", "Q02750"]

        preds = ground_columns(
            ["fam"],
            "dataset",
            "domain",
            llm_fetch=lambda _m, _c: self._openai_fixture(inner),
            reactome_client=mock_rc,
        )
        self.assertEqual(preds["fam"].ids, ["P04049", "Q02750"])
        self.assertTrue(preds["fam"].reactome_validated)


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


class ValidateIdsTests(unittest.TestCase):
    def test_validate_ids_keeps_only_resolving_protein_accessions(self) -> None:
        from reactome.client import ReactionMeta

        good = [
            ReactionMeta(st_id="R-HSA-1", display_name="rxn", species="Homo sapiens"),
        ]

        def side_effect(accession: str) -> list[ReactionMeta]:
            return good if accession == "P_GOOD" else []

        client = ReactomeClient()
        with patch.object(
            ReactomeClient, "reactions_for_protein", side_effect=side_effect
        ):
            ref = EntityRef("protein", ["P_GOOD", "P_BAD"], "fam")
            self.assertEqual(client.validate_ids(ref), ["P_GOOD"])

    def test_validate_ids_swallows_request_errors_per_accession(self) -> None:
        import httpx

        from reactome.client import ReactionMeta

        good = [
            ReactionMeta(st_id="R-HSA-1", display_name="rxn", species="Homo sapiens"),
        ]

        def side_effect(accession: str) -> list[ReactionMeta]:
            if accession == "P_TIMEOUT":
                raise httpx.ConnectTimeout("simulated")
            if accession == "P_GOOD":
                return good
            return []

        client = ReactomeClient()
        with patch.object(
            ReactomeClient, "reactions_for_protein", side_effect=side_effect
        ):
            ref = EntityRef("protein", ["P_TIMEOUT", "P_GOOD", "P_BAD"], "fam")
            self.assertEqual(client.validate_ids(ref), ["P_GOOD"])

    def test_validate_ids_metabolite_returns_all_or_none(self) -> None:
        from reactome.client import ReactionMeta

        rxn = [
            ReactionMeta(st_id="R-HSA-1", display_name="rxn", species="Homo sapiens"),
        ]
        client = ReactomeClient()

        with patch.object(ReactomeClient, "reactions_for_metabolite", return_value=rxn):
            ref = EntityRef("metabolite", ["CHEBI:18348", "CHEBI:58456"], "PIP2")
            self.assertEqual(client.validate_ids(ref), ["CHEBI:18348", "CHEBI:58456"])

        with patch.object(ReactomeClient, "reactions_for_metabolite", return_value=[]):
            ref = EntityRef("metabolite", ["CHEBI:18348", "CHEBI:58456"], "unknown")
            self.assertEqual(client.validate_ids(ref), [])


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


class ResolveGeneToUniprotsTests(unittest.TestCase):
    def _seed_search_fixture(
        self, cache_dir: Path, gene: str, payload: dict[str, Any]
    ) -> None:
        _seed_cache(
            cache_dir,
            "/search/query",
            {"query": gene, "species": "Homo sapiens", "types": "Protein"},
            payload,
        )

    def test_resolves_canonical_uniprot_and_dedupes(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            self._seed_search_fixture(
                cache_dir,
                "RAF1",
                {
                    "results": [
                        {
                            "typeName": "Protein",
                            "entries": [
                                {
                                    "databaseName": "UniProt",
                                    "referenceIdentifier": "P04049",
                                    "referenceName": '<span class="x">RAF1</span>',
                                    "species": ["Homo sapiens"],
                                },
                                {
                                    "databaseName": "UniProt",
                                    "referenceIdentifier": "P04049",
                                    "referenceName": "RAF1",
                                    "species": ["Homo sapiens"],
                                },
                            ],
                        },
                    ],
                },
            )
            client = ReactomeClient(cache_dir=cache_dir)
            with patch("reactome.client.httpx.get") as mocked_get:
                self.assertEqual(client.resolve_gene_to_uniprots("RAF1"), ["P04049"])
                mocked_get.assert_not_called()

    def test_filters_non_uniprot_databases_and_other_species(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            self._seed_search_fixture(
                cache_dir,
                "AKT1",
                {
                    "results": [
                        {
                            "typeName": "Protein",
                            "entries": [
                                {  # different DB cross-ref → drop
                                    "databaseName": "ENSEMBL",
                                    "referenceIdentifier": "ENSP_X",
                                    "referenceName": "AKT1",
                                    "species": ["Homo sapiens"],
                                },
                                {  # mouse ortholog → drop
                                    "databaseName": "UniProt",
                                    "referenceIdentifier": "P31750",
                                    "referenceName": "Akt1",
                                    "species": ["Mus musculus"],
                                },
                                {  # canonical hit
                                    "databaseName": "UniProt",
                                    "referenceIdentifier": "P31749",
                                    "referenceName": "AKT1",
                                    "species": ["Homo sapiens"],
                                },
                            ],
                        },
                    ],
                },
            )
            client = ReactomeClient(cache_dir=cache_dir)
            self.assertEqual(client.resolve_gene_to_uniprots("AKT1"), ["P31749"])

    def test_filters_by_exact_reference_name_match(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            self._seed_search_fixture(
                cache_dir,
                "MAPK8",
                {
                    "results": [
                        {
                            "typeName": "Protein",
                            "entries": [
                                {  # fuzzy hit on MAPK8IP1 — drop, name mismatch
                                    "databaseName": "UniProt",
                                    "referenceIdentifier": "Q9UQF2",
                                    "referenceName": "MAPK8IP1",
                                    "species": ["Homo sapiens"],
                                },
                                {  # canonical hit
                                    "databaseName": "UniProt",
                                    "referenceIdentifier": "P45983",
                                    "referenceName": "MAPK8",
                                    "species": ["Homo sapiens"],
                                },
                            ],
                        },
                    ],
                },
            )
            client = ReactomeClient(cache_dir=cache_dir)
            self.assertEqual(client.resolve_gene_to_uniprots("MAPK8"), ["P45983"])


class GroundColumnsResolverIntegrationTests(unittest.TestCase):
    def _openai_fixture(self, content: str) -> dict[str, object]:
        return {
            "id": "fixture",
            "model": "fixture-model",
            "choices": [{"message": {"content": content}}],
        }

    def test_gene_names_replace_hallucinated_uniprots(self) -> None:
        # Simulate the Qwen failure mode: gene names are correct (MAPK8 etc.),
        # accessions are real-but-unrelated proteins.
        inner = json.dumps(
            {
                "pjnk": {
                    "kind": "family",
                    "ids": ["Q7Z6Z7", "P45985", "Q9UCL0"],  # all wrong proteins
                    "canonical_name": "JNK family",
                    "gene_names": ["MAPK8", "MAPK9", "MAPK10"],  # correct
                    "confidence": 0.95,
                    "reasoning": "JNKs",
                },
            },
        )

        mock_rc = MagicMock(spec=ReactomeClient)

        def resolve(gname: str) -> list[str]:
            return {
                "MAPK8": ["P45983"],
                "MAPK9": ["P45984"],
                "MAPK10": ["P53779"],
            }.get(gname, [])

        mock_rc.resolve_gene_to_uniprots.side_effect = resolve
        # All three resolved ids validate (real Reactome proteins).
        mock_rc.validate_ids.return_value = ["P45983", "P45984", "P53779"]

        preds = ground_columns(
            ["pjnk"],
            "dataset",
            "domain",
            llm_fetch=lambda _m, _c: self._openai_fixture(inner),
            reactome_client=mock_rc,
        )
        self.assertEqual(sorted(preds["pjnk"].ids), ["P45983", "P45984", "P53779"])
        self.assertTrue(preds["pjnk"].reactome_validated)
        # Original LLM gene_names preserved on the artefact for audit.
        self.assertEqual(preds["pjnk"].gene_names, ["MAPK8", "MAPK9", "MAPK10"])

    def test_falls_back_to_llm_ids_when_no_gene_names(self) -> None:
        # Metabolite path: LLM ChEBIs are reliable, no gene_names, must not
        # invoke the resolver.
        inner = json.dumps(
            {
                "PIP2": {
                    "kind": "metabolite",
                    "ids": ["CHEBI:18348", "CHEBI:58456"],
                    "canonical_name": "PIP2",
                    "gene_names": [],
                    "confidence": 0.99,
                    "reasoning": "x",
                },
            },
        )
        mock_rc = MagicMock(spec=ReactomeClient)
        mock_rc.validate_ids.return_value = ["CHEBI:18348", "CHEBI:58456"]

        preds = ground_columns(
            ["PIP2"],
            "dataset",
            "domain",
            llm_fetch=lambda _m, _c: self._openai_fixture(inner),
            reactome_client=mock_rc,
        )
        mock_rc.resolve_gene_to_uniprots.assert_not_called()
        self.assertEqual(sorted(preds["PIP2"].ids), ["CHEBI:18348", "CHEBI:58456"])

    def test_falls_back_to_llm_ids_when_resolution_empty(self) -> None:
        inner = json.dumps(
            {
                "x": {
                    "kind": "protein",
                    "ids": ["P04049"],
                    "canonical_name": "RAF1",
                    "gene_names": ["MYSTERY_GENE"],
                    "confidence": 0.9,
                    "reasoning": "x",
                },
            },
        )
        mock_rc = MagicMock(spec=ReactomeClient)
        mock_rc.resolve_gene_to_uniprots.return_value = []
        mock_rc.validate_ids.return_value = ["P04049"]

        preds = ground_columns(
            ["x"],
            "dataset",
            "domain",
            llm_fetch=lambda _m, _c: self._openai_fixture(inner),
            reactome_client=mock_rc,
        )
        # Resolution returned nothing → fall back to LLM's accessions.
        self.assertEqual(preds["x"].ids, ["P04049"])


if __name__ == "__main__":
    unittest.main()
