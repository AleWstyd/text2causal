import networkx as nx
import matplotlib.pyplot as plt

from constraints.constraint_builder import PriorKnowledge


def apply_post_hoc_edits(
    graph: nx.DiGraph,
    edges_to_add: list[tuple[str, str]],
    edges_to_forbid: list[tuple[str, str]],
    *,
    on_cycle: str = "drop",
) -> tuple[nx.DiGraph, list[tuple[str, str]]]:
    """Apply post-hoc constraint edits to a discovered graph.

    Returns (edited_graph, dropped_due_to_cycle).

    Forbidden edges are removed first (this can never introduce a cycle).
    Required edges are added in input order. If adding an edge would
    create a cycle, the edge is dropped and recorded. The reverse edge
    is removed first if it exists, mirroring the existing
    apply_constraints logic.

    The function operates on a copy of `graph` and returns a new
    DiGraph; the input is not mutated.
    """
    if on_cycle not in {"drop", "raise"}:
        raise ValueError(f"on_cycle must be 'drop' or 'raise', got {on_cycle!r}")

    g = graph.copy()
    dropped_due_to_cycle: list[tuple[str, str]] = []

    for cause, effect in edges_to_forbid:
        if g.has_edge(cause, effect):
            g.remove_edge(cause, effect)

    for cause, effect in edges_to_add:
        if cause == effect:
            raise ValueError(f"Self-loop required edge is not allowed: {cause!r}")

        if g.has_edge(effect, cause):
            g.remove_edge(effect, cause)

        if g.has_edge(cause, effect):
            continue

        g.add_edge(cause, effect)
        if not nx.is_directed_acyclic_graph(g):
            g.remove_edge(cause, effect)
            if on_cycle == "raise":
                raise ValueError(
                    f"Required edge {cause} -> {effect} would create a cycle"
                )
            dropped_due_to_cycle.append((cause, effect))

    return g, dropped_due_to_cycle


def apply_constraints(graph, prior_knowledge: PriorKnowledge):
    for cause, effect in prior_knowledge.forbidden_edges:
        if graph.has_edge(cause, effect):
            graph.remove_edge(cause, effect)

    for cause, effect in prior_knowledge.required_edges:
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
