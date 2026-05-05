"""Tests for :mod:`utils.load_data` Sachs gold loaders."""

from __future__ import annotations

import unittest

from utils.load_data import (
    available_sachs_gold_versions,
    load_sachs_dataset,
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


if __name__ == "__main__":
    unittest.main()
