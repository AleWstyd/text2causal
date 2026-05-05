"""Greedy interventional equivalence search (GIES), Hauser & Bühlmann JMLR 2012."""

from __future__ import annotations

from typing import Sequence

import gies
import networkx as nx
import numpy as np

from constraints.constraint_builder import PriorKnowledge


def _adjacency_to_cpdag_digraph(
    adj: np.ndarray, variable_names: list[str]
) -> nx.DiGraph:
    """Match :func:`causal_discovery.run_ges.run_ges` mutual-arc CPDAG encoding."""

    graph = nx.DiGraph()
    graph.add_nodes_from(variable_names)
    n = len(variable_names)
    for i in range(n):
        for j in range(i + 1, n):
            aij, aji = adj[i, j], adj[j, i]
            u, v = variable_names[i], variable_names[j]
            if aij != 0.0 and aji != 0.0:
                graph.add_edge(u, v)
                graph.add_edge(v, u)
            elif aij != 0.0 and aji == 0.0:
                graph.add_edge(u, v)
            elif aji != 0.0 and aij == 0.0:
                graph.add_edge(v, u)
    return graph


def run_gies(
    data: np.ndarray,
    variable_names: list[str],
    *,
    block_sizes: Sequence[int],
    intervention_targets: Sequence[Sequence[int]],
    prior_knowledge: PriorKnowledge | None = None,
) -> nx.DiGraph:
    """Run GIES on Gaussian interventional data.

    ``prior_knowledge`` is accepted for API parity with
    :func:`causal_discovery.run_ges.run_ges` (ignored here); callers apply
    post-hoc edits after discovery.
    """

    del prior_knowledge

    x = np.asarray(data, dtype=float)
    if x.ndim != 2:
        raise ValueError("data must be a 2D matrix")
    p = x.shape[1]
    if len(variable_names) != p:
        raise ValueError("variable_names length must match data columns")

    blocks = [int(b) for b in block_sizes]
    if sum(blocks) != x.shape[0]:
        raise ValueError(f"block_sizes sum {sum(blocks)} != n_rows {x.shape[0]}")
    if len(intervention_targets) != len(blocks):
        raise ValueError("intervention_targets length must match block_sizes")

    envs: list[np.ndarray] = []
    offset = 0
    for b in blocks:
        envs.append(x[offset : offset + b])
        offset += b

    i_list: list[list[int]] = [
        sorted({int(i) for i in tup}) for tup in intervention_targets
    ]

    adj, _score = gies.fit_bic(envs, i_list)
    return _adjacency_to_cpdag_digraph(adj, variable_names)


def run_gies_naive_pooled(
    data: np.ndarray,
    variable_names: list[str],
    prior_knowledge: PriorKnowledge | None = None,
) -> nx.DiGraph:
    """Single-environment GIES baseline (all rows pooled, no interventions)."""

    n = int(np.asarray(data).shape[0])
    return run_gies(
        data,
        variable_names,
        block_sizes=(n,),
        intervention_targets=((),),
        prior_knowledge=prior_knowledge,
    )
