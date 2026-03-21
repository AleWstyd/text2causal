def build_constraints(relations, threshold=0.7):

    required_edges = []

    for r in relations:
        cause = r["cause"]
        effect = r["effect"]
        confidence = r["confidence"]

        if confidence >= threshold:
            required_edges.append((cause, effect))

    return required_edges
