from causallearn.search.ScoreBased.GES import ges
import networkx as nx


def run_ges(data, variable_names):

    record = ges(data, score_func="local_score_BIC")

    G = record["G"]

    graph = nx.DiGraph()

    adjacency = G.graph

    n = len(variable_names)

    for i in range(n):
        for j in range(n):
            if adjacency[i, j] == 1:
                graph.add_edge(variable_names[i], variable_names[j])

    return graph
