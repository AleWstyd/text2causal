from pathlib import Path
from typing import Any, Literal

import networkx as nx
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SACHS_MOOIJ2020_GML = _REPO_ROOT / "data" / "sachs" / "gold_mooij2020.gml"
_SACHS_INTERVENTIONAL_MANIFEST = (
    _REPO_ROOT / "data" / "sachs" / "interventional" / "manifest.json"
)

# Canonical Sachs column order (matches ``cdt.data.load_dataset('sachs')``).
SACHS_CANONICAL_COLUMNS: tuple[str, ...] = (
    "praf",
    "pmek",
    "plcg",
    "PIP2",
    "PIP3",
    "p44/42",
    "pakts473",
    "PKA",
    "PKC",
    "P38",
    "pjnk",
)

# Zenodo / Science CSV header names → pipeline columns.
_SACHS_RAW_TO_CANONICAL: dict[str, str] = {
    "Raf": "praf",
    "Mek": "pmek",
    "Plcg": "plcg",
    "PIP2": "PIP2",
    "PIP3": "PIP3",
    "Erk": "p44/42",
    "Akt": "pakts473",
    "PKA": "PKA",
    "PKC": "PKC",
    "P38": "P38",
    "Jnk": "pjnk",
}


def load_lucas_dataset(path: str) -> pd.DataFrame:
    """Load the LUCAS dataset."""
    feature_names = [
        "Smoking",
        "Yellow_Fingers",
        "Anxiety",
        "Peer_Pressure",
        "Genetics",
        "Attention_Disorder",
        "Born_an_Even_Day",
        "Car_Accident",
        "Fatigue",
        "Allergy",
        "Coughing",
    ]
    lucas_data = pd.read_csv(
        f"{path}/lucas0_train.data", sep=" ", names=feature_names, index_col=False
    )
    lucas_target = pd.read_csv(
        f"{path}/lucas0_train.targets", names=["Lung_Cancer"], index_col=False
    ).replace(-1, 0)
    return pd.concat([lucas_data, lucas_target], axis=1)


def available_sachs_gold_versions() -> list[str]:
    """Registered Sachs evaluation gold graphs (see ``data/sachs/``)."""
    return ["original", "mooij2020"]


def load_sachs_dataset(
    gold_version: Literal["original", "mooij2020"] = "original",
) -> tuple[pd.DataFrame, nx.DiGraph]:
    """Load the Sachs protein-signalling dataset and a ground-truth DAG.

    The historical default retains CDT's bundled Sachs consensus DAG (``gold_version='original'``).
    ``mooij2020`` loads :file:`data/sachs/gold_mooij2020.gml` (Mooij et al. 2020, Sec. 5.8).
    """
    from cdt.data import load_dataset

    data, true_graph = load_dataset("sachs")
    if gold_version == "original":
        return data, true_graph
    if gold_version == "mooij2020":
        alt = nx.read_gml(_SACHS_MOOIJ2020_GML, label="label")
        cols = {str(c) for c in data.columns}
        nodes = {str(n) for n in alt.nodes()}
        if cols != nodes:
            raise ValueError(
                f"mooij2020 gold nodes {sorted(nodes)!r} != data columns {sorted(cols)!r}"
            )
        return data, alt
    raise ValueError(
        f"Unknown gold_version {gold_version!r}; expected one of {available_sachs_gold_versions()!r}"
    )


def load_sachs_interventional_conditions() -> list[dict[str, Any]]:
    """Return the Sachs interventional manifest (nine experimental contexts).

    Each entry includes ``gies_target_indices``: indices into
    :data:`SACHS_CANONICAL_COLUMNS` for :func:`gies.fit_bic`'s ``I`` list.
    """

    import json

    raw = json.loads(_SACHS_INTERVENTIONAL_MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError("interventional manifest must be a JSON array")
    return list(raw)


def load_sachs_interventional_dataset(
    gold_version: Literal["original", "mooij2020"] = "original",
) -> tuple[pd.DataFrame, np.ndarray, nx.DiGraph]:
    """Load pooled Sachs interventional measurements with per-row targets.

    Returns the nine Zenodo/Science conditions concatenated in manifest order.
    The second return value gives, for each row, the primary pharmacological
    target as an index into :data:`SACHS_CANONICAL_COLUMNS`, or ``-1`` when
    no single-column perturbation is modeled.

    GIES splits environments using :func:`sachs_interventional_block_sizes` and
    :func:`sachs_interventional_gies_targets`; two contexts with label ``-1`` remain **separate**
    latent experiments even though the per-row index duplicates.
    """

    manifest = load_sachs_interventional_conditions()
    frames: list[pd.DataFrame] = []
    indicators: list[int] = []

    for entry in manifest:
        path = (
            _REPO_ROOT / "data" / "sachs" / "interventional" / str(entry["csv_relpath"])
        )
        df = pd.read_csv(path)
        df = df.rename(columns=_SACHS_RAW_TO_CANONICAL)
        missing = set(SACHS_CANONICAL_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        df = df[list(SACHS_CANONICAL_COLUMNS)]
        n = int(entry["n_samples"])
        if len(df) != n:
            raise ValueError(
                f"Row count mismatch for {entry['condition_id']}: "
                f"manifest says {n}, file has {len(df)}"
            )
        frames.append(df)

        tgt = entry.get("intervention_target")
        if tgt is None:
            indicators.extend([-1] * n)
        else:
            if str(tgt) not in SACHS_CANONICAL_COLUMNS:
                raise ValueError(
                    f"Unknown intervention_target {tgt!r} in manifest entry "
                    f"{entry.get('condition_id')}"
                )
            vi = SACHS_CANONICAL_COLUMNS.index(str(tgt))
            indicators.extend([vi] * n)

    data = pd.concat(frames, ignore_index=True)
    _, true_graph = load_sachs_dataset(gold_version=gold_version)
    return data, np.asarray(indicators, dtype=np.int32), true_graph


def sachs_interventional_block_sizes() -> list[int]:
    """Sample counts per manifest row (GIES environment order)."""

    return [int(x["n_samples"]) for x in load_sachs_interventional_conditions()]


def sachs_interventional_gies_targets(
    manifest: list[dict[str, Any]] | None = None,
) -> list[list[int]]:
    """Intervention targets per environment for :func:`gies.fit_bic` (``I``)."""

    m = manifest if manifest is not None else load_sachs_interventional_conditions()
    return [list(x["gies_target_indices"]) for x in m]


def load_dream4_psn_dataset(
    data_path: str = "data/dream4_psn/data.csv",
    graph_path: str = "data/dream4_psn/ground_truth.gml",
) -> tuple[pd.DataFrame, nx.DiGraph]:
    """Load the Step 7 DREAM4-PSN fallback dataset.

    The official DREAM4 PSN archive requires authenticated Synapse access in
    this environment, so Step 7 stores a Reactome/literature-derived synthetic
    signalling SCM under ``data/dream4_psn``.
    """

    data = pd.read_csv(data_path)
    graph = nx.read_gml(graph_path, label="label")
    return data, graph


def load_liverdream_dataset(
    data_path: str = "data/liverdream/data.csv",
    graph_path: str = "data/liverdream/ground_truth.gml",
) -> tuple[pd.DataFrame, nx.DiGraph]:
    """Load the real public LiverDREAM / CellNOpt signalling benchmark."""

    data = pd.read_csv(data_path)
    graph = nx.read_gml(graph_path, label="label")
    return data, graph
