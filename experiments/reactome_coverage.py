"""Coverage probe for the Reactome REST client on the Sachs dataset.

Step 2 deliverable: for each Sachs node, does ``ReactomeClient`` return at
least one reaction? For each unordered pair of Sachs nodes, does
``get_reaction_context`` return at least one role-typed record?

The temporary hand-coded grounding below will be superseded by Step 3's
LLM-driven variable grounding. Until then, it is enough to validate that
the REST client and cache pipeline can in principle resolve the Sachs
biology.

Run with ``task reactome-coverage``. The first run hits the network and
populates ``cache/reactome/``; subsequent runs replay from cache.
"""

from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path
from statistics import mean

from reactome.client import EntityRef, ReactomeClient
from utils.load_data import load_sachs_dataset

RESULT_PATH = Path("experiments/reactome_coverage.json")

SACHS_GROUNDING: dict[str, EntityRef] = {
    "praf": EntityRef("protein", ["P04049"], "RAF1"),
    "pmek": EntityRef("protein", ["Q02750", "P36507"], "MAP2K1/MAP2K2"),
    "plcg": EntityRef("protein", ["P19174", "P16885"], "PLCG1/PLCG2"),
    "PIP2": EntityRef(
        "metabolite",
        ["CHEBI:58456", "CHEBI:18348"],
        "PIP2",
    ),
    "PIP3": EntityRef(
        "metabolite",
        ["CHEBI:57836", "CHEBI:16618"],
        "PIP3",
    ),
    "p44/42": EntityRef("protein", ["P28482", "P27361"], "MAPK1/MAPK3 (ERK1/2)"),
    "pakts473": EntityRef("protein", ["P31749", "P31751", "Q9Y243"], "AKT1/2/3"),
    "PKA": EntityRef(
        "protein",
        [
            "P17612",
            "P22694",
            "P22612",
            "P10644",
            "P31321",
            "P13861",
            "P31323",
        ],
        "PKA (catalytic + regulatory subunits)",
    ),
    "PKC": EntityRef(
        "protein",
        [
            "P17252",
            "P05771",
            "P05129",
            "Q05655",
            "Q02156",
            "P24723",
            "P41743",
            "Q04759",
            "Q05513",
        ],
        "PKC (pan-isoform)",
    ),
    "P38": EntityRef(
        "protein",
        ["Q16539", "Q15759", "P53778", "O15264"],
        "p38 (MAPK11/12/13/14)",
    ),
    "pjnk": EntityRef(
        "protein",
        ["P45983", "P45984", "P53779"],
        "JNK1/2/3 (MAPK8/9/10)",
    ),
}


def _format_fraction(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator}"


def run_coverage() -> dict[str, object]:
    data, true_graph = load_sachs_dataset()
    columns = list(data.columns)
    ground_truth_pairs: set[tuple[str, str]] = {
        tuple(sorted([u, v])) for u, v in true_graph.edges()
    }

    missing_grounding = [c for c in columns if c not in SACHS_GROUNDING]
    if missing_grounding:
        raise RuntimeError(
            "Coverage script is missing a temporary grounding for: "
            f"{missing_grounding}. Update SACHS_GROUNDING in this file."
        )

    client = ReactomeClient()

    started = time.time()

    reactions_per_node: dict[str, int] = {}
    nodes_missing: list[str] = []
    reaction_metas: dict[str, list[str]] = {}
    for column in columns:
        ref = SACHS_GROUNDING[column]
        if ref.kind == "protein":
            metas = []
            for accession in ref.normalised_ids:
                metas.extend(client.reactions_for_protein(accession))
        else:
            metas = client.reactions_for_metabolite(ref.display_name)
        unique_ids = sorted({m.st_id for m in metas if m.st_id})
        reaction_metas[column] = unique_ids
        reactions_per_node[column] = len(unique_ids)
        if not unique_ids:
            nodes_missing.append(column)

    pairs = list(combinations(columns, 2))
    pairs_with_context = 0
    pairs_missing: list[list[str]] = []
    records_per_covered_pair: list[int] = []
    covered_unordered: set[tuple[str, str]] = set()
    for col_a, col_b in pairs:
        a = SACHS_GROUNDING[col_a]
        b = SACHS_GROUNDING[col_b]
        records = client.get_reaction_context(a, b)
        if records:
            pairs_with_context += 1
            records_per_covered_pair.append(len(records))
            covered_unordered.add(tuple(sorted([col_a, col_b])))
        else:
            pairs_missing.append([col_a, col_b])

    gt_covered = ground_truth_pairs & covered_unordered
    gt_missing = sorted(ground_truth_pairs - covered_unordered)

    elapsed = time.time() - started
    cache_dir = client.cache_dir
    cache_files_total = (
        sum(1 for _ in cache_dir.glob("*.json")) if cache_dir.exists() else 0
    )

    coverage: dict[str, object] = {
        "dataset": "sachs",
        "node_coverage": _format_fraction(
            len(columns) - len(nodes_missing), len(columns)
        ),
        "pair_coverage": _format_fraction(pairs_with_context, len(pairs)),
        "ground_truth_edge_coverage": _format_fraction(
            len(gt_covered), len(ground_truth_pairs)
        ),
        "mean_reactions_per_node": (
            float(mean(reactions_per_node.values())) if reactions_per_node else 0.0
        ),
        "mean_records_per_pair_with_context": (
            float(mean(records_per_covered_pair)) if records_per_covered_pair else 0.0
        ),
        "nodes_missing": nodes_missing,
        "pairs_missing": pairs_missing,
        "ground_truth_edges_missing": [list(p) for p in gt_missing],
        "notes": (
            "Reactome curates direct reaction-level participation; many Sachs "
            "edges (notably PKC- and PKA-mediated upstream regulation) are not "
            "expressed as single reactions, so reaction-level pair coverage "
            "under-represents pathway-level relevance. Step 4's LLM reasoner "
            "will see only the 'covered' pairs as Reactome context."
        ),
        "wall_clock_seconds": round(elapsed, 2),
        "cache_files_total": cache_files_total,
    }
    return coverage


def main() -> None:
    coverage = run_coverage()
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(coverage, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(coverage, indent=2, sort_keys=True))
    print(f"\nWrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
