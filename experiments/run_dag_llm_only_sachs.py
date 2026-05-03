#!/usr/bin/env python3
"""Step 4 Phase 4: emit the C-LLM-only DAG from cached LLM priors.

Reads ``experiments/causal_priors_sachs.json`` (Phase 3 output) and writes
``experiments/predicted_dag_llm_only_sachs.gml`` plus the sidecar
``experiments/predicted_dag_llm_only_sachs.meta.json``. The DAG contains
all 11 Sachs variables as nodes (some may be isolated).
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from reasoning.dag_from_priors import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    build_meta,
    predict_dag_from_priors_with_meta,
)
from utils.load_data import load_sachs_dataset

PRIORS_PATH = Path("experiments/causal_priors_sachs.json")
GML_PATH = Path("experiments/predicted_dag_llm_only_sachs.gml")
META_PATH = Path("experiments/predicted_dag_llm_only_sachs.meta.json")


def main() -> int:
    priors = json.loads(PRIORS_PATH.read_text(encoding="utf-8"))
    data, _ = load_sachs_dataset()
    variable_names = list(data.columns)

    result = predict_dag_from_priors_with_meta(
        priors,
        variable_names,
        confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    )
    meta = build_meta(result)

    GML_PATH.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gml(result.graph, GML_PATH)
    META_PATH.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(
        json.dumps(
            {
                "threshold": meta["threshold"],
                "n_claims_in_priors": meta["n_claims_in_priors"],
                "n_above_threshold": meta["n_above_threshold"],
                "n_after_constraint_filter": meta["n_after_constraint_filter"],
                "n_claims_kept": meta["n_claims_kept"],
                "n_dropped_due_to_cycle": meta["n_dropped_due_to_cycle"],
                "node_count": meta["node_count"],
                "edge_count": meta["edge_count"],
                "is_acyclic": meta["is_acyclic"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"\nWrote {GML_PATH}")
    print(f"Wrote {META_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
