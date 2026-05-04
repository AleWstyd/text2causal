"""Tests for :mod:`causal_discovery.run_lingam`."""

from __future__ import annotations

import random
import unittest
from pathlib import Path

import networkx as nx
import numpy as np

from causal_discovery.run_lingam import run_lingam
from constraints.constraint_builder import ConstraintBuilder, load_priors
from utils.load_data import load_sachs_dataset


class RunLingamOracleEncodingTests(unittest.TestCase):
    def test_oracle_dense_prior_overconstrained_forbidden_succeeds(self) -> None:
        data_df, _true_graph = load_sachs_dataset()
        names = list(data_df.columns)
        data = data_df.to_numpy()
        oracle_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "oracle_priors_sachs.json"
        )
        oracle = load_priors(oracle_path)
        builder = ConstraintBuilder(oracle, names, 0.5)
        dense = builder.build_lingam_prior_matrix()
        forbidden_only = builder.build_lingam_sparse_prior_matrix(
            lingam_required_mode="none",
            forbid_reverse_of_required=True,
        )
        np.random.seed(42)
        random.seed(42)
        with self.assertRaises(RuntimeError) as ctx:
            run_lingam(
                data,
                names,
                prior_matrix=dense,
                apply_prior_knowledge_softly=False,
            )
        self.assertIn("over-constrained", str(ctx.exception).lower())

        np.random.seed(42)
        random.seed(42)
        g = run_lingam(
            data,
            names,
            prior_matrix=forbidden_only,
            apply_prior_knowledge_softly=True,
        )
        self.assertTrue(nx.is_directed_acyclic_graph(g))


if __name__ == "__main__":
    unittest.main()
