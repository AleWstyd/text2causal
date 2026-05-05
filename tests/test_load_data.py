"""Tests for :mod:`utils.load_data` Sachs gold loaders."""

from __future__ import annotations

import unittest

from utils.load_data import (
    SACHS_CANONICAL_COLUMNS,
    available_sachs_gold_versions,
    load_sachs_interventional_conditions,
    load_sachs_interventional_dataset,
    load_sachs_dataset,
    sachs_interventional_block_sizes,
    sachs_interventional_gies_targets,
)


class LoadSachsGoldTests(unittest.TestCase):
    def test_available_versions(self) -> None:
        self.assertEqual(
            available_sachs_gold_versions(),
            ["original", "mooij2020"],
        )

    def test_original_vs_mooij2020_same_nodes_differing_edges(self) -> None:
        df_o, g_o = load_sachs_dataset("original")
        df_m, g_m = load_sachs_dataset("mooij2020")
        self.assertTrue(df_o.equals(df_m))
        self.assertEqual(set(g_o.nodes()), set(g_m.nodes()))
        self.assertEqual(set(g_o.nodes()), {str(c) for c in df_o.columns})
        e_o = set(g_o.edges())
        e_m = set(g_m.edges())
        self.assertNotEqual(e_o, e_m)
        self.assertEqual(g_o.number_of_edges(), 18)
        self.assertEqual(g_m.number_of_edges(), 19)

    def test_original_idempotent(self) -> None:
        _, a = load_sachs_dataset("original")
        _, b = load_sachs_dataset("original")
        self.assertEqual(set(a.edges()), set(b.edges()))


class LoadSachsInterventionalTests(unittest.TestCase):
    def test_manifest_matches_observational_columns(self) -> None:
        df_obs, _ = load_sachs_dataset("original")
        df_int, ind, g = load_sachs_interventional_dataset("original")
        self.assertEqual(list(df_obs.columns), list(df_int.columns))
        self.assertEqual(list(df_int.columns), list(SACHS_CANONICAL_COLUMNS))
        self.assertEqual(set(g.nodes()), {str(c) for c in df_int.columns})
        self.assertEqual(len(df_int), sum(sachs_interventional_block_sizes()))
        self.assertEqual(ind.shape[0], len(df_int))
        for v in ind.tolist():
            self.assertIn(v, range(-1, 11))

    def test_conditions_and_gies_family_length(self) -> None:
        conds = load_sachs_interventional_conditions()
        self.assertEqual(len(conds), 9)
        self.assertEqual(len(sachs_interventional_block_sizes()), 9)
        self.assertEqual(len(sachs_interventional_gies_targets()), 9)


if __name__ == "__main__":
    unittest.main()
