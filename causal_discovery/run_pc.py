from causallearn.search.ConstraintBased.PC import pc
import networkx as nx


def run_pc(data, variable_names):

    cg = pc(data, alpha=0.05, indep_test="fisherz")

    graph = nx.DiGraph()

    adjacency = cg.G.graph

    n = len(variable_names)

    for i in range(n):
        for j in range(n):
            if adjacency[i, j] == 1:
                graph.add_edge(variable_names[i], variable_names[j])

    return graph
