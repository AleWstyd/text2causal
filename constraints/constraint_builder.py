from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge


Edge = tuple[str, str]


@dataclass(frozen=True)
class PriorKnowledge:
    required_edges: list[Edge]
    forbidden_edges: list[Edge]


def _deduplicate_edges(edges: list[Edge]) -> list[Edge]:
    seen: set[Edge] = set()
    unique_edges: list[Edge] = []

    for edge in edges:
        if edge in seen:
            continue

        seen.add(edge)
        unique_edges.append(edge)

    return unique_edges


def build_prior_knowledge(
    relations: list[dict[str, object]], threshold: float = 0.7
) -> PriorKnowledge:
    required_edges: list[Edge] = []

    for relation in relations:
        cause = str(relation["cause"])
        effect = str(relation["effect"])
        confidence = float(relation["confidence"])

        if confidence >= threshold:
            required_edges.append((cause, effect))

    return PriorKnowledge(
        required_edges=_deduplicate_edges(required_edges),
        forbidden_edges=[],
    )


def validate_prior_knowledge(
    prior_knowledge: PriorKnowledge, variable_names: list[str]
) -> None:
    known_variables = set(variable_names)

    for edge_type, edges in (
        ("required", prior_knowledge.required_edges),
        ("forbidden", prior_knowledge.forbidden_edges),
    ):
        for cause, effect in edges:
            if cause == effect:
                raise ValueError(
                    f"{edge_type.title()} edge cannot be a self-loop: {cause}"
                )

            unknown_variables = {
                variable
                for variable in (cause, effect)
                if variable not in known_variables
            }
            if unknown_variables:
                unknown_list = ", ".join(sorted(unknown_variables))
                raise ValueError(
                    f"{edge_type.title()} edge references unknown variables: "
                    f"{cause} -> {effect} ({unknown_list})"
                )

    conflicting_edges = set(prior_knowledge.required_edges) & set(
        prior_knowledge.forbidden_edges
    )
    if conflicting_edges:
        conflicts = ", ".join(
            f"{cause} -> {effect}" for cause, effect in sorted(conflicting_edges)
        )
        raise ValueError(
            f"Prior knowledge contains edges marked as both required and forbidden: {conflicts}"
        )


def build_pc_background_knowledge(
    prior_knowledge: PriorKnowledge,
) -> BackgroundKnowledge:
    background_knowledge = BackgroundKnowledge()

    for cause, effect in prior_knowledge.required_edges:
        background_knowledge.add_required_by_node(GraphNode(cause), GraphNode(effect))

    for cause, effect in prior_knowledge.forbidden_edges:
        background_knowledge.add_forbidden_by_node(GraphNode(cause), GraphNode(effect))

    return background_knowledge


def build_lingam_prior_knowledge(
    prior_knowledge: PriorKnowledge, variable_names: list[str]
) -> np.ndarray:
    n_features = len(variable_names)
    lingam_prior = np.full((n_features, n_features), -1, dtype=int)
    feature_to_index = {
        feature_name: index for index, feature_name in enumerate(variable_names)
    }

    for cause, effect in prior_knowledge.required_edges:
        lingam_prior[feature_to_index[cause], feature_to_index[effect]] = 1

    for cause, effect in prior_knowledge.forbidden_edges:
        lingam_prior[feature_to_index[cause], feature_to_index[effect]] = 0

    return lingam_prior
