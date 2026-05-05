"""Tests for :mod:`causal_discovery.run_pc`."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

from causal_discovery.run_pc import run_pc
from constraints.constraint_builder import PriorKnowledge


class _FakeCG:
    """Minimal stand-in for causallearn PC output (adjacency uses PC edge convention)."""

    def __init__(self, adj: np.ndarray) -> None:
        class _G:
            pass

        self.G = _G()
        self.G.graph = adj


def _adj_with_edges(n: int, directed_pairs: list[tuple[int, int]]) -> np.ndarray:
    adj = np.zeros((n, n), dtype=int)
    for i, j in directed_pairs:
        adj[i, j] = -1
        adj[j, i] = 1
    return adj


def _adj_with_undirected(n: int, undirected_pairs: list[tuple[int, int]]) -> np.ndarray:
    adj = np.zeros((n, n), dtype=int)
    for i, j in undirected_pairs:
        adj[i, j] = -1
        adj[j, i] = -1
    return adj


class RunPcInjectionTests(unittest.TestCase):
    def test_inject_false_skips_post_hoc_even_with_prior(self) -> None:
        names = ["X0", "X1", "X2"]
        data = np.zeros((10, 3), dtype=float)
        adj = np.zeros((3, 3), dtype=int)
        pk = PriorKnowledge(required_edges=[("X0", "X2")], forbidden_edges=[])
        with patch("causal_discovery.run_pc.pc", return_value=_FakeCG(adj)):
            g, dropped = run_pc(data, names, pk, inject_required_edges=False)
        self.assertEqual(list(g.edges()), [])
        self.assertEqual(dropped, [])

    def test_inject_true_restores_required_edge_skeleton_dropped(self) -> None:
        names = ["X0", "X1", "X2"]
        data = np.zeros((10, 3), dtype=float)
        adj = np.zeros((3, 3), dtype=int)
        pk = PriorKnowledge(required_edges=[("X0", "X2")], forbidden_edges=[])
        with patch("causal_discovery.run_pc.pc", return_value=_FakeCG(adj)):
            g, dropped = run_pc(data, names, pk, inject_required_edges=True)
        self.assertIn(("X0", "X2"), g.edges())
        self.assertEqual(dropped, [])

    def test_cycle_closing_required_edge_dropped_and_reported(self) -> None:
        names = ["X0", "X1", "X2"]
        data = np.zeros((10, 3), dtype=float)
        adj = _adj_with_edges(3, [(0, 1), (1, 2)])
        pk = PriorKnowledge(required_edges=[("X2", "X0")], forbidden_edges=[])
        with patch("causal_discovery.run_pc.pc", return_value=_FakeCG(adj)):
            g, dropped = run_pc(data, names, pk, inject_required_edges=True)
        self.assertTrue(nx.is_directed_acyclic_graph(g))
        self.assertEqual(dropped, [("X2", "X0")])
        self.assertFalse(g.has_edge("X2", "X0"))


class RunPcUndirectedTests(unittest.TestCase):
    def test_preserves_undirected_as_mutual_arcs(self) -> None:
        names = ["X0", "X1", "X2"]
        data = np.zeros((10, 3), dtype=float)
        adj = _adj_with_undirected(3, [(0, 1)])
        with patch("causal_discovery.run_pc.pc", return_value=_FakeCG(adj)):
            g, dropped = run_pc(data, names, None)
        self.assertEqual(dropped, [])
        self.assertTrue(g.has_edge("X0", "X1") and g.has_edge("X1", "X0"))

    def test_post_hoc_orients_mutual_arc_to_required_direction(self) -> None:
        names = ["X0", "X1", "X2"]
        data = np.zeros((10, 3), dtype=float)
        adj = _adj_with_undirected(3, [(0, 1)])
        pk = PriorKnowledge(required_edges=[("X0", "X1")], forbidden_edges=[])
        with patch("causal_discovery.run_pc.pc", return_value=_FakeCG(adj)):
            g, dropped = run_pc(data, names, pk, inject_required_edges=True)
        self.assertEqual(dropped, [])
        self.assertTrue(g.has_edge("X0", "X1"))
        self.assertFalse(g.has_edge("X1", "X0"))


if __name__ == "__main__":
    unittest.main()
