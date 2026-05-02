"""Reactome additive evidence layer tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from llm.cache import cache_key
from reactome.client import (
    EntityRef,
    ReactionRecord,
    ReactomeClient,
    format_context_for_llm,
)


def _seed_cache(
    cache_dir: Path, path: str, params: dict[str, object], payload: object
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key({"path": path, "params": params})
    (cache_dir / f"{key}.json").write_text(json.dumps(payload), encoding="utf-8")


def _reaction_meta(st_id: str) -> list[dict[str, str]]:
    return [
        {
            "stId": st_id,
            "displayName": f"{st_id} display",
            "speciesName": "Homo sapiens",
        }
    ]


def _seed_mapping(cache_dir: Path, accession: str, reaction_ids: list[str]) -> None:
    _seed_cache(
        cache_dir,
        f"/data/mapping/UniProt/{accession}/reactions",
        {},
        [meta for st_id in reaction_ids for meta in _reaction_meta(st_id)],
    )


def _seed_details(
    cache_dir: Path,
    st_id: str,
    payload: dict[str, object],
) -> None:
    _seed_cache(cache_dir, f"/data/query/enhanced/{st_id}", {}, payload)


def _seed_participants(
    cache_dir: Path,
    st_id: str,
    payload: list[dict[str, object]],
) -> None:
    _seed_cache(cache_dir, f"/data/participants/{st_id}", {}, payload)


class ReactomeEvidenceLayerTests(unittest.TestCase):
    def test_pathways_for_entity_unions_event_of(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            _seed_mapping(cache_dir, "PAAAAA", ["R-HSA-X", "R-HSA-Y"])
            _seed_details(
                cache_dir,
                "R-HSA-X",
                {
                    "stId": "R-HSA-X",
                    "displayName": "reaction X",
                    "eventOf": [
                        {
                            "stId": "R-HSA-PATH-X",
                            "displayName": "Pathway X",
                            "schemaClass": "Pathway",
                        }
                    ],
                },
            )
            _seed_details(
                cache_dir,
                "R-HSA-Y",
                {
                    "stId": "R-HSA-Y",
                    "displayName": "reaction Y",
                    "eventOf": [
                        {
                            "stId": "R-HSA-PATH-Y",
                            "displayName": "Pathway Y",
                            "schemaClass": "TopLevelPathway",
                        }
                    ],
                },
            )

            client = ReactomeClient(cache_dir=cache_dir)
            with patch("reactome.client.httpx.get") as mocked_get:
                pathways = client.pathways_for_entity(
                    EntityRef("protein", ["PAAAAA"], "A")
                )
                mocked_get.assert_not_called()

        self.assertEqual(pathways, {"R-HSA-PATH-X", "R-HSA-PATH-Y"})

    def test_co_pathway_records_emits_one_per_shared_pathway(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            _seed_mapping(cache_dir, "PAAAAA", ["R-HSA-A"])
            _seed_mapping(cache_dir, "PBBBBB", ["R-HSA-B"])
            for st_id in ("R-HSA-A", "R-HSA-B"):
                _seed_details(
                    cache_dir,
                    st_id,
                    {
                        "stId": st_id,
                        "displayName": st_id,
                        "eventOf": [
                            {
                                "stId": "R-HSA-SHARED",
                                "displayName": "Shared pathway",
                                "schemaClass": "Pathway",
                            }
                        ],
                    },
                )

            client = ReactomeClient(cache_dir=cache_dir)
            records = client.co_pathway_records(
                EntityRef("protein", ["PAAAAA"], "A"),
                EntityRef("protein", ["PBBBBB"], "B"),
            )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].evidence_layer, "co_pathway")
        self.assertEqual(records[0].reaction_type, "Pathway")
        self.assertEqual(records[0].reaction_id, "R-HSA-SHARED")

    def test_regulator_chain_record_via_negative_regulation(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            _seed_mapping(cache_dir, "PAAAAA", [])
            _seed_mapping(cache_dir, "PBBBBB", ["R-HSA-CHAIN"])
            _seed_chain_reaction(cache_dir)

            client = ReactomeClient(cache_dir=cache_dir)
            records = client.regulator_chain_records(
                EntityRef("protein", ["PAAAAA"], "A"),
                EntityRef("protein", ["PBBBBB"], "B"),
            )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].sign, "negative")
        self.assertEqual(records[0].role_a, "regulator")
        self.assertEqual(records[0].role_b, "output")
        self.assertEqual(records[0].evidence_layer, "regulator_chain")

    def test_regulator_chain_dedupes_against_reaction_layer(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            _seed_mapping(cache_dir, "PAAAAA", ["R-HSA-CHAIN"])
            _seed_mapping(cache_dir, "PBBBBB", ["R-HSA-CHAIN"])
            _seed_chain_reaction(cache_dir)

            client = ReactomeClient(cache_dir=cache_dir)
            records = client.regulator_chain_records(
                EntityRef("protein", ["PAAAAA"], "A"),
                EntityRef("protein", ["PBBBBB"], "B"),
            )

        self.assertEqual(records, [])

    def test_co_complex_records_emit_when_two_entities_share_complex(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            _seed_mapping(cache_dir, "P04049", ["R-HSA-COMPLEX"])
            _seed_mapping(cache_dir, "Q02750", [])
            _seed_participants(
                cache_dir,
                "R-HSA-COMPLEX",
                [
                    {
                        "peDbId": 3000,
                        "displayName": "RAF:MEK complex",
                        "schemaClass": "Complex",
                        "refEntities": [
                            {"identifier": "P04049"},
                            {"identifier": "Q02750"},
                        ],
                    }
                ],
            )
            _seed_details(
                cache_dir,
                3000,
                {
                    "stId": "R-HSA-CPLX",
                    "displayName": "RAF:MEK complex",
                    "schemaClass": "Complex",
                },
            )

            client = ReactomeClient(cache_dir=cache_dir)
            records = client.co_complex_records(
                EntityRef("protein", ["P04049"], "RAF1"),
                EntityRef("protein", ["Q02750"], "MEK1"),
            )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].evidence_layer, "co_complex")
        self.assertEqual(records[0].reaction_type, "ComplexMembership")
        self.assertEqual(records[0].reaction_id, "R-HSA-CPLX")

    def test_get_evidence_records_concatenates_layers_in_order(self) -> None:
        client = ReactomeClient()
        a = EntityRef("protein", ["PAAAAA"], "A")
        b = EntityRef("protein", ["PBBBBB"], "B")

        with (
            patch.object(
                client,
                "get_reaction_context",
                return_value=[_record("reaction")],
            ),
            patch.object(
                client,
                "co_pathway_records",
                return_value=[_record("co_pathway")],
            ),
            patch.object(
                client,
                "regulator_chain_records",
                return_value=[_record("regulator_chain")],
            ),
            patch.object(
                client,
                "co_complex_records",
                return_value=[_record("co_complex")],
            ),
        ):
            records = client.get_evidence_records(a, b)

        self.assertEqual(
            [record.evidence_layer for record in records],
            ["reaction", "co_pathway", "regulator_chain", "co_complex"],
        )

    def test_format_context_for_llm_branches_per_evidence_layer(self) -> None:
        rendered = format_context_for_llm(
            [
                _record("reaction"),
                _record("co_pathway"),
                _record("regulator_chain"),
                _record("co_complex"),
            ],
            "A",
            "B",
        )

        self.assertIn("both participate in the Reactome pathway", rendered)
        self.assertIn("terminal reaction", rendered)
        self.assertIn("co-occur as members of the Reactome Complex", rendered)
        self.assertNotIn("A is an input in reaction R-HSA-co_pathway", rendered)
        self.assertNotIn("A is an input in reaction R-HSA-co_complex", rendered)


def _seed_chain_reaction(cache_dir: Path) -> None:
    _seed_details(
        cache_dir,
        "R-HSA-CHAIN",
        {
            "stId": "R-HSA-CHAIN",
            "displayName": "A negatively regulates production of B",
            "schemaClass": "Reaction",
            "output": [{"dbId": 20, "displayName": "B product"}],
            "regulatedBy": [
                {
                    "schemaClass": "NegativeRegulation",
                    "regulator": {"dbId": 10, "displayName": "A regulator"},
                }
            ],
            "eventOf": [
                {
                    "stId": "R-HSA-CHAIN-PATH",
                    "displayName": "Chain pathway",
                    "schemaClass": "Pathway",
                }
            ],
        },
    )
    _seed_participants(
        cache_dir,
        "R-HSA-CHAIN",
        [
            {"peDbId": 10, "refEntities": [{"identifier": "PAAAAA"}]},
            {"peDbId": 20, "refEntities": [{"identifier": "PBBBBB"}]},
        ],
    )


def _record(layer: str) -> ReactionRecord:
    return ReactionRecord(
        reaction_id=f"R-HSA-{layer}",
        reaction_name=f"{layer} name",
        pathway=f"{layer} pathway",
        role_a="regulator" if layer == "regulator_chain" else "input",
        role_b="output" if layer == "regulator_chain" else "input",
        reaction_type="Reaction" if layer != "co_complex" else "ComplexMembership",
        sign="negative" if layer == "regulator_chain" else None,
        evidence_layer=layer,  # type: ignore[arg-type]
    )


if __name__ == "__main__":
    unittest.main()
