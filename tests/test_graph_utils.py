"""Tests for :func:`~utils.graph_utils.apply_post_hoc_edits` (Step 5 Phase 2)."""

from __future__ import annotations

import unittest
from pathlib import Path

import networkx as nx

from constraints.constraint_builder import ConstraintBuilder, load_priors
from utils.graph_utils import apply_post_hoc_edits

_FIX_BUILDER = Path(__file__).resolve().parent / "fixtures" / "builder"


class ApplyPostHocEditsTests(unittest.TestCase):
    def test_happy_path_tree_extension(self) -> None:
        g = nx.DiGraph()
        g.add_edges_from([("A", "B")])
        out, dropped = apply_post_hoc_edits(g, [("B", "C")], [])
        self.assertEqual(dropped, [])
        self.assertTrue(nx.is_directed_acyclic_graph(out))
        self.assertEqual(set(out.edges()), {("A", "B"), ("B", "C")})

    def test_cycle_drop_closes_loop(self) -> None:
        g = nx.DiGraph()
        g.add_edges_from([("A", "B"), ("B", "C")])
        out, dropped = apply_post_hoc_edits(g, [("C", "A")], [])
        self.assertEqual(dropped, [("C", "A")])
        self.assertTrue(nx.is_directed_acyclic_graph(out))
        self.assertEqual(set(out.edges()), {("A", "B"), ("B", "C")})

    def test_order_first_adds_second_dropped(self) -> None:
        g = nx.DiGraph()
        g.add_edges_from([("A", "B"), ("B", "C")])
        out, dropped = apply_post_hoc_edits(g, [("D", "A"), ("C", "D")], [])
        self.assertEqual(dropped, [("C", "D")])
        self.assertTrue(nx.is_directed_acyclic_graph(out))
        self.assertIn(("D", "A"), out.edges())
        self.assertNotIn(("C", "D"), out.edges())

    def test_required_already_present_noop_for_that_edge(self) -> None:
        g = nx.DiGraph()
        g.add_edges_from([("A", "B")])
        out, dropped = apply_post_hoc_edits(g, [("A", "B")], [])
        self.assertEqual(dropped, [])
        self.assertEqual(out.number_of_edges(), 1)

    def test_required_orients_undirected_pair(self) -> None:
        g = nx.DiGraph()
        g.add_edge("A", "B")
        g.add_edge("B", "A")
        out, dropped = apply_post_hoc_edits(g, [("A", "B")], [])
        self.assertEqual(dropped, [])
        self.assertTrue(out.has_edge("A", "B"))
        self.assertFalse(out.has_edge("B", "A"))

    def test_required_keeps_forward_when_reverse_was_wrong_direction(self) -> None:
        g = nx.DiGraph()
        g.add_edge("B", "A")
        out, dropped = apply_post_hoc_edits(g, [("A", "B")], [])
        self.assertEqual(dropped, [])
        self.assertTrue(nx.is_directed_acyclic_graph(out))
        self.assertTrue(out.has_edge("A", "B"))
        self.assertFalse(out.has_edge("B", "A"))

    def test_forbidden_removes_edge(self) -> None:
        g = nx.DiGraph()
        g.add_edge("A", "B")
        out, dropped = apply_post_hoc_edits(g, [], [("A", "B")])
        self.assertEqual(dropped, [])
        self.assertEqual(list(out.edges()), [])

    def test_forbidden_absent_no_op(self) -> None:
        g = nx.DiGraph()
        g.add_edge("A", "B")
        out, dropped = apply_post_hoc_edits(g, [], [("X", "Y")])
        self.assertEqual(dropped, [])
        self.assertEqual(set(out.edges()), {("A", "B")})

    def test_forbid_then_add_same_endpoints(self) -> None:
        g = nx.DiGraph()
        g.add_edge("A", "B")
        out, dropped = apply_post_hoc_edits(g, [("A", "B")], [("A", "B")])
        self.assertEqual(dropped, [])
        self.assertTrue(out.has_edge("A", "B"))

    def test_on_cycle_raise_includes_endpoints(self) -> None:
        g = nx.DiGraph()
        g.add_edges_from([("A", "B"), ("B", "C")])
        with self.assertRaises(ValueError) as ctx:
            apply_post_hoc_edits(g, [("C", "A")], [], on_cycle="raise")
        self.assertEqual(
            str(ctx.exception),
            "Required edge C -> A would create a cycle",
        )

    def test_self_loop_required_raises(self) -> None:
        g = nx.DiGraph()
        g.add_node("A")
        with self.assertRaisesRegex(ValueError, "Self-loop"):
            apply_post_hoc_edits(g, [("A", "A")], [])

    def test_input_graph_not_mutated(self) -> None:
        g = nx.DiGraph()
        g.add_edges_from([("A", "B"), ("B", "C")])
        original_edges = set(g.edges())
        apply_post_hoc_edits(g, [("C", "A")], [])
        self.assertEqual(set(g.edges()), original_edges)

    def test_empty_edits_returns_distinct_copy(self) -> None:
        g = nx.DiGraph()
        g.add_edge("A", "B")
        out, dropped = apply_post_hoc_edits(g, [], [])
        self.assertEqual(dropped, [])
        self.assertIsNot(out, g)
        self.assertEqual(set(out.edges()), set(g.edges()))

    def test_end_to_end_with_constraint_builder_fixture(self) -> None:
        priors = load_priors(_FIX_BUILDER / "claims_array_one.json")
        b = ConstraintBuilder(priors, ["a", "b"], confidence_threshold=0.5)
        to_add, to_forbid = b.build_ges_post_hoc_edits()
        g = nx.DiGraph()
        g.add_nodes_from(["a", "b"])
        out, dropped = apply_post_hoc_edits(g, to_add, to_forbid)
        self.assertTrue(nx.is_directed_acyclic_graph(out))
        self.assertEqual(dropped, [])
        self.assertTrue(out.has_edge("a", "b"))
