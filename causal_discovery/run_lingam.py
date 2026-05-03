from causallearn.search.FCMBased import lingam
import networkx as nx
import numpy as np

from constraints.constraint_builder import (
    PriorKnowledge,
    build_lingam_prior_knowledge,
)


def _is_lingam_empty_candidate_error(exc: BaseException) -> bool:
    """True when causal-learn's order search exhausted the candidate set."""
    if not isinstance(exc, (ValueError, IndexError)):
        return False
    msg = str(exc)
    return "argmax" in msg and "empty sequence" in msg


def run_lingam(data, variable_names, prior_knowledge: PriorKnowledge | None = None):
    lingam_prior_knowledge = None
    n_features = len(variable_names)
    if prior_knowledge is not None:
        lingam_prior_knowledge = build_lingam_prior_knowledge(
            prior_knowledge, variable_names
        )
        n_required = int(np.sum(lingam_prior_knowledge == 1))
        n_forbidden = int(np.sum(lingam_prior_knowledge == 0))
    else:
        n_required = 0
        n_forbidden = 0

    model = lingam.DirectLiNGAM(prior_knowledge=lingam_prior_knowledge)

    try:
        model.fit(data)
    except (ValueError, IndexError) as exc:
        if not _is_lingam_empty_candidate_error(exc):
            raise
        if prior_knowledge is not None:
            raise RuntimeError(
                "LiNGAM prior matrix over-constrained: causal-learn DirectLiNGAM cannot "
                "complete the causal-order search because the candidate set was empty at "
                "some step. This is a known DirectLiNGAM limitation under dense priors; "
                f"the prior matrix has {n_required} required edges and {n_forbidden} "
                f"forbidden edges across {n_features} features. See dev_plan_v2.md "
                "Risk #6 / Step 5 LiNGAM caveat."
            ) from exc
        raise RuntimeError(
            "LiNGAM DirectLiNGAM failed during causal-order search; data may be "
            "degenerate (collinear features?)."
        ) from exc

    adjacency = model.adjacency_matrix_

    graph = nx.DiGraph()
    graph.add_nodes_from(variable_names)

    n = len(variable_names)

    for i in range(n):
        for j in range(n):
            if abs(adjacency[i, j]) > 0.001:
                graph.add_edge(variable_names[j], variable_names[i])

    return graph
