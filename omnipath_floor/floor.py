"""C0.5 ablation: build no-LLM floor priors from OmniPath direct edges.

The floor is intentionally *honest*: many OmniPath sources do not carry
direction. After ``directed=True`` filtering, a pair may be empty even when
its underlying database has an undirected edge. That is the floor's correct
answer; the higher-recall LLM path runs in parallel.

Family aggregation rule (mirrors ``ReactomeClient`` family handling): a
directed edge from any UniProt accession in ``grounding[var_a].ids`` to any
accession in ``grounding[var_b].ids`` counts as the family-level edge
``var_a → var_b``. The list of contributing primary sources is unioned
across all member-pair rows.

Confidence formula (pinned in ``docs/step_04_causal_reasoning.md``):
``min(1.0, n_sources / 3.0)``. The constraint-type label is then assigned
via ``reasoning._constraint_rule.decide_constraint_type`` so the LLM and
floor priors share identical thresholds.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from itertools import permutations
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from grounding.ground import Grounding
from reasoning._constraint_rule import ConstraintType, decide_constraint_type

DEFAULT_CACHE_DIR = Path("cache/omnipath")
DEFAULT_SNAPSHOT_NAME = "all_interactions_directed_human.parquet"

SourceFilter = Literal["all", "reactome_only"]
ConsensusSign = Literal["activates", "inhibits", "conflicting", "unknown"]

REACTOME_SOURCE_TOKEN = "reactome"


@dataclass(frozen=True)
class FloorEdge:
    """A single ordered, directed floor-prior edge between two grounded vars."""

    var_a: str
    var_b: str
    cause: str
    effect: str
    confidence: float
    constraint_type: ConstraintType
    n_sources: int
    sources: tuple[str, ...]
    is_directed: bool
    consensus_sign: ConsensusSign
    member_edges: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["sources"] = list(self.sources)
        d["member_edges"] = [list(p) for p in self.member_edges]
        return d


# ----------------------------------------------------------------------
# Snapshot fetch + cache
# ----------------------------------------------------------------------


def _load_omnipath_get(
    fetcher: Callable[[], pd.DataFrame] | None = None,
) -> Callable[[], pd.DataFrame]:
    """Return a thunk producing the raw OmniPath directed interactions DataFrame.

    The default implementation imports ``omnipath.interactions.AllInteractions``
    *lazily* so module import does not require the optional dependency. Tests
    inject ``fetcher`` directly to avoid any third-party import.
    """

    if fetcher is not None:
        return fetcher

    def _default() -> pd.DataFrame:
        from omnipath.interactions import AllInteractions

        # NB: the OmniPath REST endpoint was unstable when this snapshot
        # was first taken (omnipathdb.org and no-tls.omnipathdb.org both
        # 500'd on most parameter combinations). The query signature
        # below — ``genesymbols=True`` + ``references`` field — is the
        # combination that worked end-to-end during snapshot capture. The
        # third-party ``omnipath`` package keeps its own pickle cache at
        # ~/.cache/omnipathdb; we re-snapshot the resulting DataFrame to
        # ``cache/omnipath/`` (parquet) so that the floor pipeline is
        # reproducible from this repo without depending on the upstream
        # cache directory.
        df = AllInteractions.get(
            organisms="human",
            genesymbols=True,
            fields=["sources", "references", "curation_effort"],
        )
        if "is_directed" in df.columns:
            df = df[df["is_directed"]].reset_index(drop=True)
        return df

    return _default


def fetch_directed_interactions(
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    snapshot_name: str = DEFAULT_SNAPSHOT_NAME,
    fetcher: Callable[[], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Return the cached human directed-interactions DataFrame, fetching once.

    Subsequent calls (and cross-process replays) read from the on-disk
    parquet snapshot under ``cache/omnipath/`` so the floor pipeline is
    offline-replayable. Tests pass an in-memory ``fetcher`` to bypass the
    network entirely.
    """

    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = cache_dir / snapshot_name
    if snapshot_path.exists():
        return pd.read_parquet(snapshot_path)
    df = _load_omnipath_get(fetcher)()
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(snapshot_path, index=False)
    return df


# ----------------------------------------------------------------------
# Per-pair aggregation
# ----------------------------------------------------------------------


def _split_sources(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, float) and pd.isna(raw):
        return []
    if isinstance(raw, str):
        return [s.strip() for s in raw.split(";") if s.strip()]
    if isinstance(raw, Iterable):
        out: list[str] = []
        for item in raw:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out
    return []


def _filter_sources(sources: Iterable[str], source_filter: SourceFilter) -> list[str]:
    if source_filter == "all":
        return list(sources)
    if source_filter == "reactome_only":
        return [s for s in sources if REACTOME_SOURCE_TOKEN in s.lower()]
    raise ValueError(f"Unsupported source_filter: {source_filter!r}")


def _consensus_sign(rows: pd.DataFrame) -> ConsensusSign:
    if rows.empty:
        return "unknown"
    has_stim = bool(rows.get("is_stimulation", pd.Series(dtype=bool)).any())
    has_inhib = bool(rows.get("is_inhibition", pd.Series(dtype=bool)).any())
    if has_stim and not has_inhib:
        return "activates"
    if has_inhib and not has_stim:
        return "inhibits"
    if has_stim and has_inhib:
        return "conflicting"
    return "unknown"


def _select_protein_ids(grounding: Mapping[str, Grounding], var: str) -> set[str]:
    record = grounding[var]
    if record.kind == "metabolite":
        return set()
    return set(record.ids)


def _aggregate_pair(
    df: pd.DataFrame,
    var_a: str,
    var_b: str,
    grounding: Mapping[str, Grounding],
    source_filter: SourceFilter,
) -> FloorEdge | None:
    ids_a = _select_protein_ids(grounding, var_a)
    ids_b = _select_protein_ids(grounding, var_b)
    if not ids_a or not ids_b:
        return None

    rows = df[df["source"].isin(ids_a) & df["target"].isin(ids_b)]
    if rows.empty:
        return None

    union_sources: set[str] = set()
    member_edges: set[tuple[str, str]] = set()
    for _, row in rows.iterrows():
        union_sources.update(_split_sources(row.get("sources")))
        src = row.get("source")
        tgt = row.get("target")
        if isinstance(src, str) and isinstance(tgt, str):
            member_edges.add((src, tgt))
    filtered = sorted(set(_filter_sources(union_sources, source_filter)))
    n_sources = len(filtered)
    if n_sources == 0:
        return None
    confidence = min(1.0, n_sources / 3.0)
    constraint = decide_constraint_type(confidence)
    sign = _consensus_sign(rows)
    return FloorEdge(
        var_a=var_a,
        var_b=var_b,
        cause=var_a,
        effect=var_b,
        confidence=round(confidence, 4),
        constraint_type=constraint,
        n_sources=n_sources,
        sources=tuple(filtered),
        is_directed=True,
        consensus_sign=sign,
        member_edges=tuple(sorted(member_edges)),
    )


def build_floor_priors(
    grounding: Mapping[str, Grounding],
    *,
    source_filter: SourceFilter = "all",
    interactions_df: pd.DataFrame | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    snapshot_name: str = DEFAULT_SNAPSHOT_NAME,
    fetcher: Callable[[], pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Build OmniPath floor priors for every ordered pair of grounded variables.

    Family aggregation: any member-to-member directed edge counts as the
    family-level edge. Confidence is ``min(1.0, n_sources / 3.0)`` over the
    union of OmniPath primary sources for the family-pair (filtered to
    Reactome-named tokens when ``source_filter='reactome_only'``).
    """

    if interactions_df is None:
        df = fetch_directed_interactions(
            cache_dir=cache_dir,
            snapshot_name=snapshot_name,
            fetcher=fetcher,
        )
    else:
        df = interactions_df
    if not df.empty:
        df = df[df["is_directed"]]
    columns = sorted(grounding.keys())

    edges: list[FloorEdge] = []
    skipped_metabolite_pairs: list[list[str]] = []
    no_edge_pairs: list[list[str]] = []
    for var_a, var_b in permutations(columns, 2):
        if (
            grounding[var_a].kind == "metabolite"
            or grounding[var_b].kind == "metabolite"
        ):
            skipped_metabolite_pairs.append([var_a, var_b])
            continue
        edge = _aggregate_pair(df, var_a, var_b, grounding, source_filter)
        if edge is None:
            no_edge_pairs.append([var_a, var_b])
            continue
        edges.append(edge)

    n_high_conf = sum(1 for e in edges if e.confidence >= 0.9)
    n_soft = sum(1 for e in edges if 0.6 <= e.confidence < 0.9)
    n_unknown = sum(1 for e in edges if e.confidence < 0.6)

    return {
        "schema_version": "step04.floor.v1",
        "source_filter": source_filter,
        "family_aggregation_rule": (
            "A directed OmniPath edge from any UniProt member of var_a's "
            "grounding to any UniProt member of var_b's grounding counts as "
            "the family-level edge var_a → var_b. The contributing primary "
            "sources are unioned across all member-pair rows."
        ),
        "confidence_formula": "min(1.0, n_sources / 3.0)",
        "constraint_rule": (
            "confidence >= 0.9 -> hard_required; 0.6 <= confidence < 0.9 -> "
            "soft_prior; otherwise unknown. See "
            "reasoning/_constraint_rule.py."
        ),
        "n_pairs_total": len(columns) * (len(columns) - 1),
        "n_pairs_skipped_metabolite": len(skipped_metabolite_pairs),
        "n_pairs_no_edge": len(no_edge_pairs),
        "n_pairs_with_edge": len(edges),
        "n_hard_required": n_high_conf,
        "n_soft_prior": n_soft,
        "n_unknown_kept": n_unknown,
        "skipped_metabolite_pairs": skipped_metabolite_pairs,
        "no_edge_pairs": no_edge_pairs,
        "pairs": [edge.as_dict() for edge in edges],
    }


def write_floor_priors(out_path: Path, payload: dict[str, Any]) -> None:
    """Persist a floor-priors payload as canonical-form JSON."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
