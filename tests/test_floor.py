"""Regression tests for the OmniPath C0.5 floor-priors path (Step 4 Phase 1)."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from grounding.ground import Grounding
from omnipath_floor.floor import (
    build_floor_priors,
    fetch_directed_interactions,
)
from reasoning._constraint_rule import (
    HARD_THRESHOLD,
    SOFT_THRESHOLD,
    decide_constraint_type,
)


def _ground(
    column: str,
    kind: str,
    ids: list[str],
    *,
    canonical: str | None = None,
) -> Grounding:
    return Grounding(
        column=column,
        kind=kind,  # type: ignore[arg-type]
        ids=ids,
        canonical_name=canonical or column,
        gene_names=[],
        confidence=1.0,
        reasoning="fixture",
        reactome_validated=True,
        served_model="test/model",
    )


def _interactions_df(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a DataFrame matching the AllInteractions.get column shape."""

    return pd.DataFrame(
        rows,
        columns=[
            "source",
            "target",
            "is_directed",
            "is_stimulation",
            "is_inhibition",
            "sources",
        ],
    )


class ConstraintRuleTests(unittest.TestCase):
    def test_threshold_boundaries_match_spec(self) -> None:
        self.assertEqual(decide_constraint_type(HARD_THRESHOLD), "hard_required")
        self.assertEqual(
            decide_constraint_type(HARD_THRESHOLD, forbidden_reverse=True),
            "hard_forbidden_reverse",
        )
        self.assertEqual(decide_constraint_type(0.95), "hard_required")
        self.assertEqual(decide_constraint_type(SOFT_THRESHOLD), "soft_prior")
        self.assertEqual(decide_constraint_type(0.7), "soft_prior")
        self.assertEqual(decide_constraint_type(SOFT_THRESHOLD - 0.001), "unknown")
        self.assertEqual(decide_constraint_type(0.0), "unknown")


class FloorAggregationTests(unittest.TestCase):
    def test_single_source_yields_unknown_constraint(self) -> None:
        grounding = {
            "praf": _ground("praf", "protein", ["P04049"]),
            "pmek": _ground("pmek", "protein", ["Q02750"]),
        }
        df = _interactions_df(
            [
                {
                    "source": "P04049",
                    "target": "Q02750",
                    "is_directed": True,
                    "is_stimulation": True,
                    "is_inhibition": False,
                    "sources": "Reactome",
                },
            ]
        )
        payload = build_floor_priors(grounding, interactions_df=df)
        edges = payload["pairs"]
        forward = [e for e in edges if e["cause"] == "praf" and e["effect"] == "pmek"]
        self.assertEqual(len(forward), 1)
        self.assertEqual(forward[0]["n_sources"], 1)
        self.assertAlmostEqual(forward[0]["confidence"], 1 / 3, places=4)
        self.assertEqual(forward[0]["constraint_type"], "unknown")
        self.assertEqual(forward[0]["consensus_sign"], "activates")

    def test_two_sources_promote_to_soft_prior(self) -> None:
        grounding = {
            "praf": _ground("praf", "protein", ["P04049"]),
            "pmek": _ground("pmek", "protein", ["Q02750"]),
        }
        df = _interactions_df(
            [
                {
                    "source": "P04049",
                    "target": "Q02750",
                    "is_directed": True,
                    "is_stimulation": True,
                    "is_inhibition": False,
                    "sources": "Reactome;SIGNOR",
                },
            ]
        )
        payload = build_floor_priors(grounding, interactions_df=df)
        forward = next(
            e
            for e in payload["pairs"]
            if e["cause"] == "praf" and e["effect"] == "pmek"
        )
        self.assertEqual(forward["n_sources"], 2)
        self.assertAlmostEqual(forward["confidence"], 2 / 3, places=4)
        self.assertEqual(forward["constraint_type"], "soft_prior")

    def test_three_sources_cap_at_hard_required(self) -> None:
        grounding = {
            "praf": _ground("praf", "protein", ["P04049"]),
            "pmek": _ground("pmek", "protein", ["Q02750"]),
        }
        df = _interactions_df(
            [
                {
                    "source": "P04049",
                    "target": "Q02750",
                    "is_directed": True,
                    "is_stimulation": True,
                    "is_inhibition": False,
                    "sources": "Reactome;SIGNOR;KEGG;PathwayCommons",
                },
            ]
        )
        payload = build_floor_priors(grounding, interactions_df=df)
        forward = next(
            e
            for e in payload["pairs"]
            if e["cause"] == "praf" and e["effect"] == "pmek"
        )
        self.assertEqual(forward["n_sources"], 4)
        self.assertEqual(forward["confidence"], 1.0)
        self.assertEqual(forward["constraint_type"], "hard_required")
        self.assertEqual(forward["consensus_sign"], "activates")

    def test_family_aggregation_any_member_match_counts(self) -> None:
        grounding = {
            "PKC": _ground("PKC", "family", ["P17252", "P05771", "Q05655"]),
            "praf": _ground("praf", "protein", ["P04049"]),
        }
        df = _interactions_df(
            [
                {
                    "source": "P05771",
                    "target": "P04049",
                    "is_directed": True,
                    "is_stimulation": True,
                    "is_inhibition": False,
                    "sources": "SIGNOR",
                },
                {
                    "source": "Q05655",
                    "target": "P04049",
                    "is_directed": True,
                    "is_stimulation": False,
                    "is_inhibition": False,
                    "sources": "Reactome;PhosphoSite",
                },
            ]
        )
        payload = build_floor_priors(grounding, interactions_df=df)
        forward = next(
            e for e in payload["pairs"] if e["cause"] == "PKC" and e["effect"] == "praf"
        )
        self.assertEqual(forward["n_sources"], 3)
        self.assertEqual(forward["constraint_type"], "hard_required")
        self.assertCountEqual(forward["sources"], ["SIGNOR", "Reactome", "PhosphoSite"])
        self.assertCountEqual(
            forward["member_edges"],
            [["P05771", "P04049"], ["Q05655", "P04049"]],
        )

    def test_reactome_only_filter_drops_non_reactome_sources(self) -> None:
        grounding = {
            "praf": _ground("praf", "protein", ["P04049"]),
            "pmek": _ground("pmek", "protein", ["Q02750"]),
        }
        df = _interactions_df(
            [
                {
                    "source": "P04049",
                    "target": "Q02750",
                    "is_directed": True,
                    "is_stimulation": True,
                    "is_inhibition": False,
                    "sources": "SIGNOR;PhosphoSite_MIMP",
                },
            ]
        )
        payload = build_floor_priors(
            grounding, source_filter="reactome_only", interactions_df=df
        )
        self.assertEqual(payload["pairs"], [])
        self.assertIn(["praf", "pmek"], payload["no_edge_pairs"])

    def test_reactome_only_keeps_reactome_named_sources(self) -> None:
        grounding = {
            "praf": _ground("praf", "protein", ["P04049"]),
            "pmek": _ground("pmek", "protein", ["Q02750"]),
        }
        df = _interactions_df(
            [
                {
                    "source": "P04049",
                    "target": "Q02750",
                    "is_directed": True,
                    "is_stimulation": True,
                    "is_inhibition": False,
                    "sources": "Reactome;SIGNOR;Reactome_complex",
                },
            ]
        )
        payload = build_floor_priors(
            grounding, source_filter="reactome_only", interactions_df=df
        )
        forward = next(
            e
            for e in payload["pairs"]
            if e["cause"] == "praf" and e["effect"] == "pmek"
        )
        self.assertEqual(forward["n_sources"], 2)
        self.assertCountEqual(forward["sources"], ["Reactome", "Reactome_complex"])

    def test_metabolite_pairs_skipped_with_explicit_log(self) -> None:
        grounding = {
            "PIP3": _ground("PIP3", "metabolite", ["CHEBI:16618", "CHEBI:57836"]),
            "pakts473": _ground("pakts473", "family", ["P31749", "P31751"]),
        }
        df = _interactions_df([])
        payload = build_floor_priors(grounding, interactions_df=df)
        self.assertEqual(payload["pairs"], [])
        self.assertEqual(payload["n_pairs_skipped_metabolite"], 2)
        self.assertCountEqual(
            payload["skipped_metabolite_pairs"],
            [["PIP3", "pakts473"], ["pakts473", "PIP3"]],
        )

    def test_invalid_source_filter_raises(self) -> None:
        grounding = {
            "praf": _ground("praf", "protein", ["P04049"]),
            "pmek": _ground("pmek", "protein", ["Q02750"]),
        }
        df = _interactions_df(
            [
                {
                    "source": "P04049",
                    "target": "Q02750",
                    "is_directed": True,
                    "is_stimulation": True,
                    "is_inhibition": False,
                    "sources": "Reactome",
                },
            ]
        )
        with self.assertRaises(ValueError):
            build_floor_priors(
                grounding,
                source_filter="bogus",  # type: ignore[arg-type]
                interactions_df=df,
            )


class FetchDirectedInteractionsTests(unittest.TestCase):
    def test_snapshot_caches_dataframe_and_avoids_refetch(self) -> None:
        calls = {"count": 0}
        rows = pd.DataFrame(
            [
                {
                    "source": "P04049",
                    "target": "Q02750",
                    "is_directed": True,
                    "is_stimulation": True,
                    "is_inhibition": False,
                    "sources": "Reactome",
                }
            ]
        )

        def fake_fetch() -> pd.DataFrame:
            calls["count"] += 1
            return rows

        with TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            df = fetch_directed_interactions(cache_dir=cache_dir, fetcher=fake_fetch)
            self.assertEqual(len(df), 1)
            self.assertEqual(calls["count"], 1)

            df2 = fetch_directed_interactions(cache_dir=cache_dir, fetcher=fake_fetch)
            self.assertEqual(len(df2), 1)
            self.assertEqual(
                calls["count"],
                1,
                "Second call must replay from disk; OmniPath snapshot already cached.",
            )


if __name__ == "__main__":
    unittest.main()
