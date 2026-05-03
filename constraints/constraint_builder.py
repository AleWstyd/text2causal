from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge


Edge = tuple[str, str]

_VALID_CLAIM_CONSTRAINT_TYPES = frozenset(
    {"hard_required", "soft_prior", "hard_forbidden_reverse", "unknown", "no_context"}
)


@dataclass(frozen=True)
class ClaimRecord:
    var_a: str
    var_b: str
    cause: str  # may be "unknown"
    effect: str  # may be "unknown"
    confidence: float
    constraint_type: str  # recognised types only at filter time
    source: str  # "reactome_llm" | "omnipath_all" | "omnipath_reactome_only"


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
    forbidden_edges: list[Edge] = []

    for relation in relations:
        cause = str(relation["cause"])
        effect = str(relation["effect"])
        confidence = float(relation["confidence"])
        relation_type = str(relation.get("relation_type", "required"))

        if relation_type not in {"required", "forbidden"}:
            raise ValueError(
                "Relation type must be 'required' or 'forbidden': "
                f"{cause} -> {effect} ({relation_type})"
            )

        if confidence < threshold:
            continue

        if relation_type == "required":
            required_edges.append((cause, effect))
        else:
            forbidden_edges.append((cause, effect))

    return PriorKnowledge(
        required_edges=_deduplicate_edges(required_edges),
        forbidden_edges=_deduplicate_edges(forbidden_edges),
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


def load_priors(path: Path | str) -> list[ClaimRecord]:
    """Load Step 4 prior JSON (``claims`` or ``pairs`` array) into :class:`ClaimRecord`."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(
            f"Expected a JSON object at top level, got {type(data).__name__}"
        )

    if "claims" in data:
        raw_items = data["claims"]
    elif "pairs" in data:
        raw_items = data["pairs"]
    else:
        raise ValueError(f"Missing 'claims' or 'pairs' array in priors file: {p}")

    if not isinstance(raw_items, list):
        raise TypeError(f"'claims'/'pairs' must be a list in {p}")

    items: list[dict[str, object]] = []
    for i, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise TypeError(f"Claim {i} is not an object in {p}")
        items.append(item)

    if not items:
        return []

    if "served_models" in data or any("served_models" in c for c in items):
        source = "reactome_llm"
    elif "reactome_only" in p.name.lower():
        source = "omnipath_reactome_only"
    else:
        source = "omnipath_all"

    records: list[ClaimRecord] = []
    for item in items:
        records.append(
            ClaimRecord(
                var_a=str(item["var_a"]),
                var_b=str(item["var_b"]),
                cause=str(item["cause"]),
                effect=str(item["effect"]),
                confidence=float(item["confidence"]),
                constraint_type=str(item["constraint_type"]),
                source=source,
            )
        )
    return records


def _opposing_required_edges(required_edges: list[Edge]) -> Edge | None:
    req_set = set(required_edges)
    for u, v in required_edges:
        if (v, u) in req_set:
            return (u, v)
    return None


class ConstraintBuilder:
    """Threshold-aware prior → PC / LiNGAM / GES (post-hoc) constraint builder."""

    def __init__(
        self,
        priors: list[ClaimRecord],
        variable_names: list[str],
        confidence_threshold: float,
    ) -> None:
        self.priors = priors
        self.variable_names = variable_names
        self.confidence_threshold = confidence_threshold

    def _raise_bad_constraint_type(self, claim: ClaimRecord) -> None:
        raise ValueError(
            f"Unrecognised constraint_type={claim.constraint_type!r} "
            f"(var_a={claim.var_a!r}, var_b={claim.var_b!r})"
        )

    def _accounting_bucket(self, claim: ClaimRecord) -> str:
        if claim.constraint_type not in _VALID_CLAIM_CONSTRAINT_TYPES:
            self._raise_bad_constraint_type(claim)

        if claim.constraint_type in {"unknown", "no_context"}:
            return "discarded_unknown"
        if claim.cause == "unknown" or claim.effect == "unknown":
            return "discarded_unknown"
        if claim.confidence < self.confidence_threshold:
            return "discarded_below_threshold"

        if claim.constraint_type in {"hard_required", "soft_prior"}:
            return "kept_required"
        if claim.constraint_type == "hard_forbidden_reverse":
            return "kept_forbidden"
        self._raise_bad_constraint_type(claim)

    def summary(self) -> dict[str, int]:
        """Counts by category at the current threshold; every prior is accounted for."""
        total = len(self.priors)
        out: dict[str, int] = {
            "total": total,
            "hard_required": 0,
            "soft_prior": 0,
            "hard_forbidden_reverse": 0,
            "unknown": 0,
            "no_context": 0,
            "discarded_below_threshold": 0,
            "discarded_unknown": 0,
            "kept_required": 0,
            "kept_forbidden": 0,
        }

        for claim in self.priors:
            ctype = claim.constraint_type
            if ctype not in _VALID_CLAIM_CONSTRAINT_TYPES:
                self._raise_bad_constraint_type(claim)
            if ctype == "hard_required":
                out["hard_required"] += 1
            elif ctype == "soft_prior":
                out["soft_prior"] += 1
            elif ctype == "hard_forbidden_reverse":
                out["hard_forbidden_reverse"] += 1
            elif ctype == "unknown":
                out["unknown"] += 1
            elif ctype == "no_context":
                out["no_context"] += 1

            bucket = self._accounting_bucket(claim)
            out[bucket] += 1

        type_sum = (
            out["hard_required"]
            + out["soft_prior"]
            + out["hard_forbidden_reverse"]
            + out["unknown"]
            + out["no_context"]
        )
        if type_sum != total:
            raise RuntimeError(
                f"Internal error: constraint_type counts sum to {type_sum}, not {total}"
            )

        flow_sum = (
            out["kept_required"]
            + out["kept_forbidden"]
            + out["discarded_below_threshold"]
            + out["discarded_unknown"]
        )
        if flow_sum != total:
            raise RuntimeError(
                f"Internal error: flow buckets sum to {flow_sum}, not {total}"
            )

        return out

    def filtered_priors(self) -> tuple[list[Edge], list[Edge]]:
        """Return ``(required_edges, forbidden_edges)`` at the current threshold."""
        required_map: dict[Edge, tuple[str, str]] = {}
        forbidden_map: dict[Edge, tuple[str, str]] = {}

        for claim in self.priors:
            if claim.constraint_type not in _VALID_CLAIM_CONSTRAINT_TYPES:
                self._raise_bad_constraint_type(claim)

            if (
                claim.constraint_type in {"unknown", "no_context"}
                or claim.cause == "unknown"
                or claim.effect == "unknown"
            ):
                continue
            if claim.confidence < self.confidence_threshold:
                continue

            if claim.constraint_type in {"hard_required", "soft_prior"}:
                edge = (claim.cause, claim.effect)
                if edge in forbidden_map:
                    va, vb = forbidden_map[edge]
                    raise ValueError(
                        "Same ordered edge is both required and forbidden "
                        f"(var_a={claim.var_a!r}, var_b={claim.var_b!r}; "
                        f"conflicts with claim var_a={va!r}, var_b={vb!r})"
                    )
                required_map.setdefault(edge, (claim.var_a, claim.var_b))
            elif claim.constraint_type == "hard_forbidden_reverse":
                edge = (claim.effect, claim.cause)
                if edge in required_map:
                    va, vb = required_map[edge]
                    raise ValueError(
                        "Same ordered edge is both required and forbidden "
                        f"(var_a={claim.var_a!r}, var_b={claim.var_b!r}; "
                        f"conflicts with claim var_a={va!r}, var_b={vb!r})"
                    )
                forbidden_map.setdefault(edge, (claim.var_a, claim.var_b))
            else:
                self._raise_bad_constraint_type(claim)

        return (list(required_map.keys()), list(forbidden_map.keys()))

    def to_prior_knowledge(self) -> PriorKnowledge:
        required, forbidden = self.filtered_priors()
        opposing = _opposing_required_edges(required)
        if opposing is not None:
            u, v = opposing
            raise ValueError(
                f"Prior knowledge contains opposing required edges: {u} -> {v} and {v} -> {u}"
            )
        pk = PriorKnowledge(
            required_edges=required,
            forbidden_edges=forbidden,
        )
        validate_prior_knowledge(pk, self.variable_names)
        return pk

    def build_pc_background_knowledge(self) -> BackgroundKnowledge:
        return build_pc_background_knowledge(self.to_prior_knowledge())

    def build_lingam_prior_matrix(self) -> np.ndarray:
        return build_lingam_prior_knowledge(
            self.to_prior_knowledge(),
            self.variable_names,
        )

    def build_ges_post_hoc_edits(self) -> tuple[list[Edge], list[Edge]]:
        """Return ``(edges_to_add, edges_to_forbid)`` for post-hoc GES editing."""
        return self.filtered_priors()
