from pathlib import Path
from typing import Literal

import networkx as nx
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SACHS_MOOIJ2020_GML = _REPO_ROOT / "data" / "sachs" / "gold_mooij2020.gml"


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
