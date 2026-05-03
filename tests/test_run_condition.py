"""Tests for :mod:`experiments.run_condition` (Step 5 Phase 3)."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

from constraints.constraint_builder import ClaimRecord
from experiments import run_condition as run_condition_mod
from experiments.run_condition import _derive_condition_label, run_condition


def _synth_chain_data(
    rng: np.random.Generator, *, n_samples: int = 800
) -> tuple[np.ndarray, list[str], nx.DiGraph]:
    variable_names = ["X0", "X1", "X2"]
    x0 = rng.normal(size=n_samples)
    x1 = 0.9 * x0 + 0.2 * rng.normal(size=n_samples)
    x2 = 0.9 * x1 + 0.2 * rng.normal(size=n_samples)
    data = np.column_stack([x0, x1, x2])
    true_graph = nx.DiGraph()
    true_graph.add_edges_from([("X0", "X1"), ("X1", "X2")])
    return data, variable_names, true_graph


def _find_cycle_closing_edge(graph: nx.DiGraph) -> tuple[str, str] | None:
    """Return (y, x) such that adding y -> x completes a cycle (shortest path x ~> y uses ≥ 2 edges)."""
    for y in graph.nodes():
        for x in graph.nodes():
            if x == y or graph.has_edge(y, x):
                continue
            try:
                path = nx.shortest_path(graph, x, y)
            except nx.NetworkXNoPath:
                continue
            if len(path) - 1 < 2:
                continue
            return (y, x)
    return None


class RunConditionC0Tests(unittest.TestCase):
    def test_c0_happy_path_all_algorithms(self) -> None:
        rng = np.random.default_rng(202)
        data, names, true_graph = _synth_chain_data(rng)
        for algorithm in ("PC", "GES", "LiNGAM"):
            with self.subTest(algorithm=algorithm):
                result = run_condition(
                    dataset_name="synth_chain",
                    data=data,
                    variable_names=names,
                    true_graph=true_graph,
                    priors=None,
                    priors_source="none",
                    algorithm=algorithm,
                    threshold=None,
                    seed=1,
                )
                self.assertEqual(result["status"], "ok")
                self.assertIsNotNone(result["metrics"])
                assert result["metrics"] is not None
                for key in ("shd", "aupr", "precision", "recall", "f1"):
                    self.assertIn(key, result["metrics"])
                self.assertIsNone(result["constraint_summary"])
                self.assertEqual(result["dropped_due_to_cycle"], [])


class RunConditionOracleTests(unittest.TestCase):
    def test_oracle_lingam_low_shd(self) -> None:
        rng = np.random.default_rng(303)
        data, names, true_graph = _synth_chain_data(rng, n_samples=1200)
        result = run_condition(
            dataset_name="synth_chain",
            data=data,
            variable_names=names,
            true_graph=true_graph,
            priors=None,
            priors_source="oracle",
            algorithm="LiNGAM",
            threshold=None,
            seed=4,
        )
        self.assertEqual(result["status"], "ok")
        assert result["metrics"] is not None
        self.assertLess(result["metrics"]["shd"], 8.0)


class RunConditionReactomeThresholdTests(unittest.TestCase):
    def test_c3_kept_required_count_pc(self) -> None:
        rng = np.random.default_rng(404)
        data, names, true_graph = _synth_chain_data(rng)
        priors = [
            ClaimRecord(
                var_a="X0",
                var_b="X1",
                cause="X0",
                effect="X1",
                confidence=0.8,
                constraint_type="hard_required",
                source="reactome_llm",
            ),
            ClaimRecord(
                var_a="X1",
                var_b="X2",
                cause="X1",
                effect="X2",
                confidence=0.5,
                constraint_type="hard_required",
                source="reactome_llm",
            ),
        ]
        result = run_condition(
            dataset_name="synth_chain",
            data=data,
            variable_names=names,
            true_graph=true_graph,
            priors=priors,
            priors_source="reactome_llm",
            algorithm="PC",
            threshold=0.7,
            seed=0,
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["condition"], "C3")
        assert result["constraint_summary"] is not None
        self.assertEqual(result["constraint_summary"]["kept_required"], 1)


class RunConditionGesCycleTests(unittest.TestCase):
    def test_ges_drops_cycle_edge_stays_dag(self) -> None:
        rng = np.random.default_rng(505)
        data, names, true_graph = _synth_chain_data(rng)
        ges_stub = nx.DiGraph()
        ges_stub.add_nodes_from(names)
        ges_stub.add_edges_from([("X0", "X1"), ("X1", "X2")])
        pair = _find_cycle_closing_edge(ges_stub)
        self.assertEqual(pair, ("X2", "X0"))
        cause, effect = pair
        priors = [
            ClaimRecord(
                var_a=cause,
                var_b=effect,
                cause=cause,
                effect=effect,
                confidence=1.0,
                constraint_type="hard_required",
                source="reactome_llm",
            )
        ]
        with patch.object(run_condition_mod, "run_ges", return_value=ges_stub.copy()):
            result = run_condition(
                dataset_name="synth_chain",
                data=data,
                variable_names=names,
                true_graph=true_graph,
                priors=priors,
                priors_source="reactome_llm",
                algorithm="GES",
                threshold=0.5,
                seed=0,
            )
        self.assertEqual(result["status"], "ok")
        self.assertNotEqual(result["dropped_due_to_cycle"], [])
        pred = nx.DiGraph()
        pred.add_edges_from(tuple(edge) for edge in result["predicted_edges"])
        self.assertTrue(nx.is_directed_acyclic_graph(pred))


class RunConditionPCFailureTests(unittest.TestCase):
    def test_pc_opposing_required_fails_gracefully(self) -> None:
        rng = np.random.default_rng(606)
        data, names, true_graph = _synth_chain_data(rng)
        priors = [
            ClaimRecord(
                var_a="X0",
                var_b="X1",
                cause="X0",
                effect="X1",
                confidence=1.0,
                constraint_type="hard_required",
                source="reactome_llm",
            ),
            ClaimRecord(
                var_a="X1",
                var_b="X0",
                cause="X1",
                effect="X0",
                confidence=1.0,
                constraint_type="hard_required",
                source="reactome_llm",
            ),
        ]
        result = run_condition(
            dataset_name="synth_chain",
            data=data,
            variable_names=names,
            true_graph=true_graph,
            priors=priors,
            priors_source="reactome_llm",
            algorithm="PC",
            threshold=0.5,
            seed=0,
        )
        self.assertEqual(result["status"], "failed")
        self.assertIsNotNone(result["error"])
        self.assertTrue(result["error"])
        self.assertIsNone(result["metrics"])
        self.assertEqual(result["predicted_edges"], [])


class RunConditionValidationTests(unittest.TestCase):
    def test_none_requires_null_priors_and_threshold(self) -> None:
        rng = np.random.default_rng(707)
        data, names, true_graph = _synth_chain_data(rng)
        bad_priors = [
            ClaimRecord(
                var_a="X0",
                var_b="X1",
                cause="X0",
                effect="X1",
                confidence=1.0,
                constraint_type="hard_required",
                source="reactome_llm",
            )
        ]
        with self.assertRaises(ValueError):
            run_condition(
                dataset_name="synth_chain",
                data=data,
                variable_names=names,
                true_graph=true_graph,
                priors=bad_priors,
                priors_source="none",
                algorithm="PC",
                threshold=None,
                seed=0,
            )
        with self.assertRaises(ValueError):
            run_condition(
                dataset_name="synth_chain",
                data=data,
                variable_names=names,
                true_graph=true_graph,
                priors=None,
                priors_source="none",
                algorithm="PC",
                threshold=0.7,
                seed=0,
            )

    def test_invalid_priors_source(self) -> None:
        rng = np.random.default_rng(808)
        data, names, true_graph = _synth_chain_data(rng)
        with self.assertRaises(ValueError):
            run_condition(
                dataset_name="synth_chain",
                data=data,
                variable_names=names,
                true_graph=true_graph,
                priors=None,
                priors_source="not_a_source",
                algorithm="PC",
                threshold=None,
                seed=0,
            )

    def test_invalid_algorithm(self) -> None:
        rng = np.random.default_rng(909)
        data, names, true_graph = _synth_chain_data(rng)
        with self.assertRaises(ValueError):
            run_condition(
                dataset_name="synth_chain",
                data=data,
                variable_names=names,
                true_graph=true_graph,
                priors=None,
                priors_source="none",
                algorithm="not_algo",
                threshold=None,
                seed=0,
            )


class RunConditionDeterminismTests(unittest.TestCase):
    def test_c0_pc_same_seed_same_edges(self) -> None:
        rng = np.random.default_rng(1001)
        data, names, true_graph = _synth_chain_data(rng)
        common = dict(
            dataset_name="synth_chain",
            data=data,
            variable_names=names,
            true_graph=true_graph,
            priors=None,
            priors_source="none",
            algorithm="PC",
            threshold=None,
            seed=42,
        )
        a = run_condition(**common)
        b = run_condition(**common)
        self.assertEqual(a["predicted_edges"], b["predicted_edges"])


class DeriveConditionLabelTests(unittest.TestCase):
    def test_labels(self) -> None:
        cases: list[tuple[str, float | None, str]] = [
            ("none", None, "C0"),
            ("omnipath_all", None, "C0.5"),
            ("omnipath_all", 0.2, "C0.5"),
            ("omnipath_reactome_only", None, "C0.5_reactome_only"),
            ("oracle", None, "C5"),
            ("oracle", 0.99, "C5"),
            ("reactome_llm", 0.9, "C2"),
            ("reactome_llm", 0.7, "C3"),
            ("reactome_llm", 0.6, "C4"),
            ("reactome_llm", 0.65, "C_llm_t0.65"),
        ]
        for priors_source, thr, expected in cases:
            with self.subTest(priors_source=priors_source, thr=thr):
                self.assertEqual(_derive_condition_label(priors_source, thr), expected)


class RunConditionJsonTests(unittest.TestCase):
    def test_all_sample_outputs_json_serialisable(self) -> None:
        rng = np.random.default_rng(1102)
        data, names, true_graph = _synth_chain_data(rng)
        samples: list[dict] = []

        for algorithm in ("PC", "GES", "LiNGAM"):
            samples.append(
                run_condition(
                    dataset_name="synth_chain",
                    data=data,
                    variable_names=names,
                    true_graph=true_graph,
                    priors=None,
                    priors_source="none",
                    algorithm=algorithm,
                    threshold=None,
                    seed=2,
                )
            )

        samples.append(
            run_condition(
                dataset_name="synth_chain",
                data=data,
                variable_names=names,
                true_graph=true_graph,
                priors=None,
                priors_source="oracle",
                algorithm="LiNGAM",
                threshold=None,
                seed=3,
            )
        )

        samples.append(
            run_condition(
                dataset_name="synth_chain",
                data=data,
                variable_names=names,
                true_graph=true_graph,
                priors=[
                    ClaimRecord(
                        var_a="X0",
                        var_b="X1",
                        cause="X0",
                        effect="X1",
                        confidence=0.8,
                        constraint_type="hard_required",
                        source="reactome_llm",
                    )
                ],
                priors_source="reactome_llm",
                algorithm="PC",
                threshold=0.7,
                seed=0,
            )
        )
        priors_opp = [
            ClaimRecord(
                var_a="X0",
                var_b="X1",
                cause="X0",
                effect="X1",
                confidence=1.0,
                constraint_type="hard_required",
                source="reactome_llm",
            ),
            ClaimRecord(
                var_a="X1",
                var_b="X0",
                cause="X1",
                effect="X0",
                confidence=1.0,
                constraint_type="hard_required",
                source="reactome_llm",
            ),
        ]
        samples.append(
            run_condition(
                dataset_name="synth_chain",
                data=data,
                variable_names=names,
                true_graph=true_graph,
                priors=priors_opp,
                priors_source="reactome_llm",
                algorithm="PC",
                threshold=0.5,
                seed=0,
            )
        )

        for row in samples:
            with self.subTest(status=row["status"], algorithm=row["algorithm"]):
                json.dumps(row)
