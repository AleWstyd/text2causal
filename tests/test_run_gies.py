"""Regression tests for :mod:`causal_discovery.run_gies`."""

from __future__ import annotations

import unittest

import numpy as np

from causal_discovery.run_gies import run_gies


class GiesSyntheticTests(unittest.TestCase):
    def test_four_node_chain_partial_intervention(self) -> None:
        rng = np.random.default_rng(42)
        n = 800
        # X0 -> X1 -> X2 -> X3 linear Gaussian
        e0 = rng.normal(size=n)
        e1 = rng.normal(size=n)
        e2 = rng.normal(size=n)
        e3 = rng.normal(size=n)
        x0 = e0
        x1 = 0.9 * x0 + e1
        x2 = 0.85 * x1 + e2
        x3 = 0.8 * x2 + e3
        obs = np.column_stack([x0, x1, x2, x3])
        # Second environment: hard shift on X0 (surrogate intervention context)
        x0_i = rng.normal(scale=2.0, size=n)
        x1_i = 0.9 * x0_i + e1
        x2_i = 0.85 * x1_i + e2
        x3_i = 0.8 * x2_i + e3
        intv = np.column_stack([x0_i, x1_i, x2_i, x3_i])

        names = ["X0", "X1", "X2", "X3"]
        g = run_gies(
            np.vstack([obs, intv]),
            names,
            block_sizes=(n, n),
            intervention_targets=([], [0]),
            prior_knowledge=None,
        )
        self.assertTrue(g.has_edge("X0", "X1"))
        self.assertTrue(g.has_edge("X1", "X2"))
        self.assertTrue(g.has_edge("X2", "X3"))


if __name__ == "__main__":
    unittest.main()
