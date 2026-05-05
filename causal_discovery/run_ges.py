from causallearn.search.ScoreBased.GES import ges
import networkx as nx

from constraints.constraint_builder import PriorKnowledge


def run_ges(data, variable_names, prior_knowledge: PriorKnowledge | None = None):
    del prior_knowledge

    record = ges(data, score_func="local_score_BIC", node_names=variable_names)

    G = record["G"]

    graph = nx.DiGraph()
    graph.add_nodes_from(variable_names)

    adjacency = G.graph

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

    return graph
