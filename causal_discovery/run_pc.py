from causallearn.search.ConstraintBased.PC import pc
import networkx as nx

from constraints.constraint_builder import (
    PriorKnowledge,
    build_pc_background_knowledge,
)
from utils.graph_utils import apply_post_hoc_edits

_PC_POST_HOC_META_KEY = "pc_post_hoc_required_added"


def run_pc(
    data,
    variable_names,
    prior_knowledge: PriorKnowledge | None = None,
    alpha: float = 0.05,
    *,
    inject_required_edges: bool = True,
) -> tuple[nx.DiGraph, list[tuple[str, str]]]:
    """Run PC with optional Fisher-Z independence testing and prior knowledge.

    When ``inject_required_edges`` is True and ``prior_knowledge`` is set,
    required/forbidden edges are applied *after* the PC orientation phase via
    :func:`utils.graph_utils.apply_post_hoc_edits`, matching the GES post-hoc
    pattern: native ``BackgroundKnowledge`` may not keep required edges in the
    skeleton.

    Returns ``(graph, dropped_due_to_cycle)``. The count of required edges that
    were missing from native PC but successfully injected is stored under
    ``graph.graph['pc_post_hoc_required_added']`` (int). Call
    :func:`consume_pc_post_hoc_required_added` to read and remove it.
    """
    background_knowledge = None
    if prior_knowledge is not None:
        background_knowledge = build_pc_background_knowledge(prior_knowledge)

    cg = pc(
        data,
        alpha=alpha,
        indep_test="fisherz",
        background_knowledge=background_knowledge,
        node_names=variable_names,
        show_progress=False,
    )

    graph = nx.DiGraph()
    graph.add_nodes_from(variable_names)

    adjacency = cg.G.graph

    n = len(variable_names)

    for i in range(n):
        for j in range(i + 1, n):
            aij, aji = adjacency[i, j], adjacency[j, i]
            u, v = variable_names[i], variable_names[j]
            if aij == -1 and aji == -1:
                graph.add_edge(u, v)
                graph.add_edge(v, u)
            elif aij == -1 and aji == 1:
                graph.add_edge(u, v)
            elif aij == 1 and aji == -1:
                graph.add_edge(v, u)

    graph.graph[_PC_POST_HOC_META_KEY] = 0
    dropped: list[tuple[str, str]] = []
    if inject_required_edges and prior_knowledge is not None:
        native = graph.copy()
        graph, dropped = apply_post_hoc_edits(
            graph,
            list(prior_knowledge.required_edges),
            list(prior_knowledge.forbidden_edges),
        )
        dropped_set = set(dropped)
        required_injected = sum(
            1
            for u, v in prior_knowledge.required_edges
            if (u, v) not in dropped_set
            and not native.has_edge(u, v)
            and graph.has_edge(u, v)
        )
        graph.graph[_PC_POST_HOC_META_KEY] = int(required_injected)
    return graph, dropped


def consume_pc_post_hoc_required_added(graph: nx.DiGraph) -> int:
    """Return and remove the post-hoc required-edge injection count from ``graph``."""
    return int(graph.graph.pop(_PC_POST_HOC_META_KEY, 0))
