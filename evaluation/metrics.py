def _normalize_edges(edges):
    """Normalize edges to undirected form for comparison."""
    return {tuple(sorted([u, v])) for u, v in edges}


def precision(predicted_edges, true_edges):

    predicted = _normalize_edges(predicted_edges)
    true = _normalize_edges(true_edges)

    if len(predicted) == 0:
        return 0

    correct = predicted.intersection(true)

    return len(correct) / len(predicted)


def recall(predicted_edges, true_edges):

    predicted = _normalize_edges(predicted_edges)
    true = _normalize_edges(true_edges)

    correct = predicted.intersection(true)

    return len(correct) / len(true)


def f1_score(p, r):

    if (p + r) == 0:
        return 0

    return 2 * (p * r) / (p + r)


def calculate_metrics(cg, true_graph):
    """Calculate evaluation metrics: Precision, Recall, F1-Score."""
    inferred_edges = _normalize_edges(cg["model_edges"])
    true_edges = _normalize_edges(true_graph.edges())

    tp = len(inferred_edges & true_edges)  # True positives
    fp = len(inferred_edges - true_edges)  # False positives
    fn = len(true_edges - inferred_edges)  # False negatives

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    )

    return precision, recall, f1
