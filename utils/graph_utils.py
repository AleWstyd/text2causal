import networkx as nx
import matplotlib.pyplot as plt


def apply_constraints(graph, required_edges):

    for cause, effect in required_edges:
        if graph.has_edge(effect, cause):
            graph.remove_edge(effect, cause)

        graph.add_edge(cause, effect)

    return graph


def build_hierarchical_layout(graph):
    if not graph.nodes:
        return {}

    condensation = nx.condensation(graph)
    node_to_component = condensation.graph["mapping"]
    component_layers = {}
    for layer_index, generation in enumerate(nx.topological_generations(condensation)):
        for component in generation:
            component_layers[component] = layer_index

    layers = {}
    for node in graph.nodes():
        layer = component_layers[node_to_component[node]]
        layers.setdefault(layer, []).append(node)

    positions = {}
    total_layers = max(layers) + 1 if layers else 1

    for layer_index in sorted(layers):
        nodes = sorted(layers[layer_index])
        node_count = len(nodes)

        if node_count == 1:
            x_positions = [0.0]
        else:
            x_positions = [
                -1.0 + (2.0 * index / (node_count - 1)) for index in range(node_count)
            ]

        y_position = float(total_layers - layer_index - 1)

        for node, x_position in zip(nodes, x_positions, strict=False):
            positions[node] = (x_position, y_position)

    return positions


def build_graph_figure(graph, title="Graph", pos=None):
    fig, ax = plt.subplots(figsize=(6, 6))

    if pos is None:
        pos = build_hierarchical_layout(graph)

    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=2000,
        node_color="lightblue",
        font_size=10,
        arrows=True,
        arrowsize=18,
        connectionstyle="arc3,rad=0.05",
        ax=ax,
    )

    ax.set_title(title)
    fig.tight_layout()

    return fig


def visualize_graph(graph, title="Graph"):
    fig = build_graph_figure(graph, title=title)

    plt.show()

    return fig
