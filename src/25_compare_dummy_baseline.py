
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "results" / "expanded_ecg_dataset_clean.npz"
METADATA_PATH = PROJECT_DIR / "results" / "expanded_ecg_metadata_clean.csv"
OUTPUT_PATH = PROJECT_DIR / "results" / "dummy_vs_logistic_predictions.csv"
SUMMARY_PATH = PROJECT_DIR / "results" / "dummy_vs_logistic_summary.txt"


def main():
    feature_path = PROJECT_DIR / "results" / "multirecord_ecg_features.csv"

    # The grouped baseline used the expanded dataset and its engineered
    # features. Use the saved grouped predictions' feature source only if
    # the original feature file is available and row-aligned.
    if not DATA_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("Clean dataset or metadata is missing.")
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {feature_path}\n"
            "This script needs feature rows aligned with the expanded dataset. "
            "Do not use features from a different, smaller dataset."
        )

    with np.load(DATA_PATH, allow_pickle=True) as data:
        record_ids_from_data = data["record_ids"].astype(str)

    metadata = pd.read_csv(METADATA_PATH)
    features = pd.read_csv(feature_path)

    print("Clean dataset segments:", len(metadata))
    print("Rows in feature file:", len(features))
    print("Unique recordings in dataset:", metadata["record_id"].astype(str).nunique())

    raise SystemExit(
        "Safety check: the multirecord feature file was built for the earlier "
        "small dataset, so it may not align with the 74-segment expanded "
        "dataset. No model was trained. We need to build features from the "
        "clean expanded dataset and preserve its exact row order first."
    )


if __name__ == "__main__":
    main()
