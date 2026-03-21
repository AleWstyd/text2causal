import networkx as nx
import matplotlib.pyplot as plt


def apply_constraints(graph, required_edges):

    for cause, effect in required_edges:
        if graph.has_edge(effect, cause):
            graph.remove_edge(effect, cause)

        graph.add_edge(cause, effect)

    return graph


def visualize_graph(graph, title="Graph"):

    plt.figure(figsize=(6, 6))

    pos = nx.spring_layout(graph)

    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=2000,
        node_color="lightblue",
        font_size=10,
    )

    plt.title(title)

    plt.show()
