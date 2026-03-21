from causallearn.search.FCMBased import lingam
import networkx as nx

from constraints.constraint_builder import (
    PriorKnowledge,
    build_lingam_prior_knowledge,
)


def run_lingam(data, variable_names, prior_knowledge: PriorKnowledge | None = None):
    lingam_prior_knowledge = None
    if prior_knowledge is not None:
        lingam_prior_knowledge = build_lingam_prior_knowledge(
            prior_knowledge, variable_names
        )

    model = lingam.DirectLiNGAM(prior_knowledge=lingam_prior_knowledge)

    model.fit(data)

    adjacency = model.adjacency_matrix_

    graph = nx.DiGraph()
    graph.add_nodes_from(variable_names)

    n = len(variable_names)

    for i in range(n):
        for j in range(n):
            if abs(adjacency[i, j]) > 0.001:
                graph.add_edge(variable_names[j], variable_names[i])

    return graph
