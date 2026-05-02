"""Reactome client regression tests.

The fixture suite covers cache replay, role parsing, family expansion,
metabolite search via ``/search/query``, signed regulator records, and
LLM-facing context rendering.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx

from llm.cache import cache_key
from reactome.client import (
    EntityRef,
    ReactionRecord,
    ReactomeClient,
    _normalize_chebi,
    _strip_highlight,
    format_context_for_llm,
)

FIXTURES = Path(__file__).parent / "fixtures" / "reactome"


def _load_fixture(name: str) -> object:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _seed_cache(
    cache_dir: Path, path: str, params: dict[str, object], payload: object
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key({"path": path, "params": params})
    (cache_dir / f"{key}.json").write_text(json.dumps(payload), encoding="utf-8")


class _CacheSeeder:
    """Helper that pre-populates a Reactome cache_dir from committed fixtures."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    def seed_protein_reactions(self, accession: str, fixture: str) -> None:
        _seed_cache(
            self.cache_dir,
            f"/data/mapping/UniProt/{accession}/reactions",
            {},
            _load_fixture(fixture),
        )

    def seed_query_enhanced(self, st_id: str | int, fixture: str) -> None:
        _seed_cache(
            self.cache_dir,
            f"/data/query/enhanced/{st_id}",
            {},
            _load_fixture(fixture),
        )

    def seed_participants(self, st_id: str, fixture: str) -> None:
        _seed_cache(
            self.cache_dir,
            f"/data/participants/{st_id}",
            {},
            _load_fixture(fixture),
        )

    def seed_search(self, query: str, types: str, payload: object) -> None:
        _seed_cache(
            self.cache_dir,
            "/search/query",
            {"query": query, "species": "Homo sapiens", "types": types},
            payload,
        )


class HelperTests(unittest.TestCase):
    def test_strip_highlight_removes_span_markup(self) -> None:
        raw = '<span class="highlighting">RAF1</span> kinase'
        self.assertEqual(_strip_highlight(raw), "RAF1 kinase")

    def test_normalize_chebi_handles_prefixed_and_bare(self) -> None:
        self.assertEqual(_normalize_chebi("18348"), "CHEBI:18348")
        self.assertEqual(_normalize_chebi("CHEBI:18348"), "CHEBI:18348")
        self.assertEqual(_normalize_chebi("chebi:16618"), "CHEBI:16618")
        self.assertIsNone(_normalize_chebi(None))


class ReactomeClientCacheTests(unittest.TestCase):
    def test_cache_hit_avoids_network(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            seeder = _CacheSeeder(cache_dir)
            seeder.seed_protein_reactions("P04049", "mapping_uniprot_P04049.json")

            client = ReactomeClient(cache_dir=cache_dir)
            with patch("reactome.client.httpx.get") as mocked_get:
                metas = client.reactions_for_protein("P04049")
                mocked_get.assert_not_called()

            self.assertEqual(len(metas), 1)
            self.assertEqual(metas[0].st_id, "R-HSA-5672978")
            self.assertEqual(metas[0].species, "Homo sapiens")

    def test_protein_reactions_filter_non_human_species(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            _CacheSeeder(cache_dir).seed_protein_reactions(
                "P04049", "mapping_uniprot_P04049.json"
            )
            client = ReactomeClient(cache_dir=cache_dir)
            metas = client.reactions_for_protein("P04049")
            self.assertTrue(all(m.species == "Homo sapiens" for m in metas))

    def test_reactions_for_protein_swallows_404(self) -> None:
        request = httpx.Request("GET", "https://reactome.test/missing")
        response = httpx.Response(404, request=request)
        error = httpx.HTTPStatusError(
            "not found",
            request=request,
            response=response,
        )
        client = ReactomeClient()

        with patch.object(client, "_get", side_effect=error):
            self.assertEqual(client.reactions_for_protein("NOPE"), [])

    def test_reactions_for_metabolite_filters_html_and_non_human(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            _CacheSeeder(cache_dir).seed_search(
                "PIP2",
                "Reaction",
                _load_fixture("search_query_pip2.json"),
            )
            client = ReactomeClient(cache_dir=cache_dir)

            with patch("reactome.client.httpx.get") as mocked_get:
                metas = client.reactions_for_metabolite("PIP2")
                mocked_get.assert_not_called()

        self.assertEqual(len(metas), 1)
        self.assertEqual(metas[0].st_id, "R-HSA-PIP2")
        self.assertEqual(metas[0].display_name, "PI3K phosphorylates PIP2 to PIP3")
        self.assertNotIn("<span", metas[0].display_name)
        self.assertEqual(metas[0].species, "Homo sapiens")


class RoleParsingTests(unittest.TestCase):
    def _build_client(self) -> tuple[ReactomeClient, Path]:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cache_dir = Path(tmp.name)
        seeder = _CacheSeeder(cache_dir)
        seeder.seed_protein_reactions("P04049", "mapping_uniprot_P04049.json")
        seeder.seed_protein_reactions("Q02750", "mapping_uniprot_Q02750.json")
        seeder.seed_protein_reactions("P36507", "mapping_uniprot_P36507.json")
        seeder.seed_protein_reactions("P17612", "mapping_uniprot_P17612.json")
        seeder.seed_protein_reactions("O00255", "mapping_uniprot_O00255.json")
        seeder.seed_query_enhanced("R-HSA-5672978", "query_enhanced_R-HSA-5672978.json")
        seeder.seed_query_enhanced("R-HSA-NEG", "query_enhanced_R-HSA-NEG.json")
        seeder.seed_query_enhanced(9999, "query_enhanced_9999.json")
        seeder.seed_participants("R-HSA-5672978", "participants_R-HSA-5672978.json")
        seeder.seed_participants("R-HSA-NEG", "participants_R-HSA-NEG.json")
        return ReactomeClient(cache_dir=cache_dir), cache_dir

    def test_role_assignment_for_known_reaction(self) -> None:
        client, _ = self._build_client()
        with patch("reactome.client.httpx.get") as mocked_get:
            records = client.get_reaction_context(
                EntityRef("protein", ["P04049"], "RAF1"),
                EntityRef("protein", ["Q02750"], "MAP2K1"),
            )
            mocked_get.assert_not_called()

        self.assertTrue(records)
        roles = {(r.role_a, r.role_b) for r in records}
        self.assertIn(("catalyst", "input"), roles)
        self.assertIn(("catalyst", "output"), roles)
        for record in records:
            self.assertEqual(record.reaction_id, "R-HSA-5672978")
            self.assertEqual(record.reaction_type, "Reaction")
            self.assertEqual(record.pathway, "RAF/MAP kinase cascade")

    def test_family_expansion_unions_reactions_across_isoforms(self) -> None:
        client, _ = self._build_client()
        with patch("reactome.client.httpx.get") as mocked_get:
            records = client.get_reaction_context(
                EntityRef("protein", ["P04049"], "RAF1"),
                EntityRef("protein", ["Q02750", "P36507"], "MAP2K1/MAP2K2"),
            )
            mocked_get.assert_not_called()

        self.assertTrue(records)
        supporting = {tuple(r.supporting_ids_b) for r in records}
        self.assertTrue(any("Q02750" in ids and "P36507" in ids for ids in supporting))

    def test_bare_int_regulator_resolved_via_followup(self) -> None:
        client, _ = self._build_client()
        with patch("reactome.client.httpx.get") as mocked_get:
            details = client.reaction_details("R-HSA-5672978")
            participants = client.reaction_participants("R-HSA-5672978")
            mocked_get.assert_not_called()
        regs = details.get("regulatedBy", [])
        self.assertIn(9999, regs)
        self.assertIn(1006, participants)

    def test_negative_regulator_yields_signed_record(self) -> None:
        client, _ = self._build_client()

        with patch("reactome.client.httpx.get") as mocked_get:
            records = client.get_reaction_context(
                EntityRef("protein", ["P17612"], "PKA-cat"),
                EntityRef("protein", ["Q02750"], "MEK1"),
            )
            mocked_get.assert_not_called()

        self.assertTrue(
            any(
                r.reaction_id == "R-HSA-NEG"
                and r.role_a == "regulator"
                and r.role_b == "output"
                and r.sign == "negative"
                for r in records
            )
        )

    def test_positive_metabolite_regulator_yields_signed_record(self) -> None:
        client, cache_dir = self._build_client()
        _CacheSeeder(cache_dir).seed_search(
            "PS",
            "Reaction",
            {
                "results": [
                    {
                        "typeName": "Reaction",
                        "entries": [
                            {
                                "stId": "R-HSA-5672978",
                                "name": "RAF phosphorylates MAP2K dimer",
                                "species": ["Homo sapiens"],
                            }
                        ],
                    }
                ]
            },
        )

        with patch("reactome.client.httpx.get") as mocked_get:
            records = client.get_reaction_context(
                EntityRef("metabolite", ["CHEBI:17636"], "PS"),
                EntityRef("protein", ["Q02750"], "MEK1"),
            )
            mocked_get.assert_not_called()

        self.assertTrue(
            any(
                r.role_a == "regulator"
                and r.role_b in {"input", "output"}
                and r.sign == "positive"
                for r in records
            )
        )

    def test_bare_int_regulator_yields_negative_record(self) -> None:
        client, _ = self._build_client()

        with patch("reactome.client.httpx.get") as mocked_get:
            records = client.get_reaction_context(
                EntityRef("protein", ["O00255"], "letX"),
                EntityRef("protein", ["Q02750"], "MEK1"),
            )
            mocked_get.assert_not_called()

        self.assertTrue(
            any(
                r.role_a == "regulator"
                and r.role_b in {"input", "output"}
                and r.sign == "negative"
                for r in records
            )
        )


class FormatContextTests(unittest.TestCase):
    def test_renders_paragraph_per_record(self) -> None:
        records = [
            ReactionRecord(
                reaction_id="R-HSA-5672978",
                reaction_name="RAF phosphorylates MAP2K dimer",
                pathway="RAF/MAP kinase cascade",
                role_a="catalyst",
                role_b="output",
                reaction_type="Reaction",
                supporting_ids_a=("P04049",),
                supporting_ids_b=("Q02750", "P36507"),
            )
        ]
        rendered = format_context_for_llm(records, "RAF1", "MAP2K1")
        self.assertIn("RAF/MAP kinase cascade", rendered)
        self.assertIn("R-HSA-5672978", rendered)
        self.assertIn("RAF1", rendered)
        self.assertIn("MAP2K1", rendered)
        self.assertIn("catalyst", rendered)

    def test_signed_regulator_phrase(self) -> None:
        records = [
            ReactionRecord(
                reaction_id="R-HSA-1",
                reaction_name="example",
                pathway=None,
                role_a="regulator",
                role_b="output",
                reaction_type="Reaction",
                sign="negative",
            )
        ]
        rendered = format_context_for_llm(records, "X", "Y")
        self.assertIn("negative regulator", rendered)

    def test_empty_records_yields_explicit_message(self) -> None:
        rendered = format_context_for_llm([], "X", "Y")
        self.assertIn("No Reactome reactions", rendered)


if __name__ == "__main__":
    unittest.main()
