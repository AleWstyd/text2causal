from causallearn.search.ConstraintBased.PC import pc
import networkx as nx

from constraints.constraint_builder import (
    PriorKnowledge,
    build_pc_background_knowledge,
)


def run_pc(data, variable_names, prior_knowledge: PriorKnowledge | None = None):
    background_knowledge = None
    if prior_knowledge is not None:
        background_knowledge = build_pc_background_knowledge(prior_knowledge)

    cg = pc(
        data,
        alpha=0.05,
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
        for j in range(n):
            if adjacency[i, j] == 1:
                graph.add_edge(variable_names[i], variable_names[j])

    return graph
