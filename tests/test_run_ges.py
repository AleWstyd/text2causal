"""Tests for :mod:`causal_discovery.run_ges`."""

from __future__ import annotations

import unittest
import random

import numpy as np

from causal_discovery.run_ges import run_ges
from utils.load_data import load_sachs_dataset


class RunGesSachsTests(unittest.TestCase):
    def test_c0_seed0_has_mutual_arc_for_some_cpdag_edge(self) -> None:
        df, _true = load_sachs_dataset()
        data = df.to_numpy()
        names = list(df.columns)
        np.random.seed(0)
        random.seed(0)

        g = run_ges(data, names, None)
        mutual = {(u, v) for u, v in g.edges() if u != v and g.has_edge(v, u)}
        self.assertGreater(
            len(mutual),
            0,
            "Expected at least one undirected CPDAG edge as mutual DiGraph arcs on Sachs",
        )


if __name__ == "__main__":
    unittest.main()
