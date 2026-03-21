from causallearn.search.FCMBased import lingam
import networkx as nx


def run_lingam(data, variable_names):

    model = lingam.ICALiNGAM()

    model.fit(data)

    adjacency = model.adjacency_matrix_

    graph = nx.DiGraph()

    n = len(variable_names)

    for i in range(n):
        for j in range(n):
            if abs(adjacency[i, j]) > 0.001:
                graph.add_edge(variable_names[j], variable_names[i])

    return graph
