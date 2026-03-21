import pandas as pd


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
