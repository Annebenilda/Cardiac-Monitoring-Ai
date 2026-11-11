
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)

FEATURE_PATH = Path("results/multirecord_ecg_features.csv")
DATA_PATH = Path("results/multirecord_ecg_dataset.npz")
OUTPUT_PATH = Path("results/baseline_predictions.csv")

FEATURE_COLUMNS = [
    "filtered_amplitude_range",
    "filtered_signal_std",
    "filtered_rms",
    "detected_peak_count",
    "mean_rr_seconds",
    "rr_std_seconds",
    "rr_rmssd_seconds",
    "estimated_rate_bpm",
]


def main():
    if not FEATURE_PATH.exists() or not DATA_PATH.exists():
        raise FileNotFoundError(
            "Run scripts 11 and 14 first."
        )

    df = pd.read_csv(FEATURE_PATH)
    data = np.load(DATA_PATH, allow_pickle=True)
    labels = data["labels"].astype(str)
    record_ids = data["record_ids"].astype(str)

    # Use the dataset row order as the stable segment identifier.
    if len(df) != len(labels):
        raise ValueError("Feature and dataset row counts differ.")

    df["label"] = labels
    df["record_id"] = record_ids

    X = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    y = df["label"].to_numpy()
    groups = df["record_id"].to_numpy()

    print("Segments:", len(df))
    print("Unique records:", len(np.unique(groups)))
    print("\nLabels:")
    print(pd.Series(y).value_counts())

    if len(np.unique(groups)) < 3:
        raise ValueError(
            "Need at least three records for this evaluation."
        )

    logo = LeaveOneGroupOut()
    predictions = np.empty(len(y), dtype=object)
    predictions[:] = ""

    for fold, (train_idx, test_idx) in enumerate(
        logo.split(X, y, groups), start=1
    ):
        y_train = y[train_idx]
        y_test = y[test_idx]

        # A fold must contain both classes in training.
        if len(np.unique(y_train)) < 2:
            print(f"Skipping fold {fold}: training has one class.")
            continue

        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            ),
        )

        model.fit(X.iloc[train_idx], y_train)
        predictions[test_idx] = model.predict(X.iloc[test_idx])

        print(
            f"Fold {fold}: held out record "
            f"{groups[test_idx[0]]}"
        )

    valid = predictions != ""

    if not valid.any():
        raise RuntimeError(
            "No folds were evaluated. Check labels and record groups."
        )

    y_true = y[valid]
    y_pred = predictions[valid]

    print("\n=== Exploratory leave-one-record-out results ===")
    print("Evaluated segments:", int(valid.sum()))
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 3))
    print(
        "Balanced accuracy:",
        round(balanced_accuracy_score(y_true, y_pred), 3),
    )
    print("\nConfusion matrix (labels AFIB, NON_AFIB):")
    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=["AFIB", "NON_AFIB"],
        )
    )
    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=["AFIB", "NON_AFIB"],
            zero_division=0,
        )
    )

    output = df[["record_id", "label"]].copy()
    output["predicted_label"] = predictions
    output["evaluated"] = valid
    output.to_csv(OUTPUT_PATH, index=False)

    print(f"\nPredictions saved to: {OUTPUT_PATH}")
    print(
        "\nCAUTION: Only 10 segments from 5 records were used. "
        "These results are for pipeline debugging, not clinical "
        "performance claims."
    )


if __name__ == "__main__":
    main()