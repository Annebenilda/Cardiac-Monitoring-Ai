
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks, welch

from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "results" / "expanded_ecg_dataset_clean.npz"
METADATA_PATH = PROJECT_DIR / "results" / "expanded_ecg_metadata_clean.csv"

FEATURES_PATH = PROJECT_DIR / "results" / "clean_dataset_features.csv"
PREDICTIONS_PATH = PROJECT_DIR / "results" / "dummy_vs_logistic_predictions.csv"
SUMMARY_PATH = PROJECT_DIR / "results" / "dummy_vs_logistic_summary.txt"

FS = 250


def extract_features(signal):
    """Extract exploratory signal features from one ECG segment."""
    signal = np.asarray(signal, dtype=float).reshape(-1)

    if len(signal) < FS * 2 or not np.all(np.isfinite(signal)):
        raise ValueError("Signal is too short or contains non-finite values.")

    signal = signal - np.median(signal)

    sos = butter(
        3, [0.5, 40.0],
        btype="bandpass",
        fs=FS,
        output="sos",
    )
    filtered = sosfiltfilt(sos, signal)

    abs_signal = np.abs(filtered)
    robust_range = (
        np.percentile(filtered, 95) - np.percentile(filtered, 5)
    )

    # Exploratory peak timing only: not a validated QRS detector.
    prominence = max(np.std(abs_signal) * 0.5, 1e-8)
    peaks, _ = find_peaks(
        abs_signal,
        distance=int(0.25 * FS),
        prominence=prominence,
    )

    rr = np.diff(peaks) / FS

    if len(rr):
        rr_mean = float(np.mean(rr))
        rr_median = float(np.median(rr))
        rr_std = float(np.std(rr))
        rr_rmssd = (
            float(np.sqrt(np.mean(np.diff(rr) ** 2)))
            if len(rr) > 1 else np.nan
        )
        estimated_rate = 60.0 / rr_mean if rr_mean > 0 else np.nan
    else:
        rr_mean = rr_median = rr_std = rr_rmssd = estimated_rate = np.nan

    frequencies, power = welch(filtered, fs=FS, nperseg=min(1024, len(filtered)))

    def relative_power(low, high):
        mask = (frequencies >= low) & (frequencies < high)
        total = np.sum(power)
        if not np.any(mask) or total <= 0:
            return np.nan
        return float(np.sum(power[mask]) / total)

    return {
        "signal_std": float(np.std(signal)),
        "signal_rms": float(np.sqrt(np.mean(signal ** 2))),
        "amplitude_range": float(np.ptp(signal)),
        "robust_range": float(robust_range),
        "iqr": float(
            np.percentile(signal, 75) - np.percentile(signal, 25)
        ),
        "filtered_std": float(np.std(filtered)),
        "filtered_rms": float(np.sqrt(np.mean(filtered ** 2))),
        "filtered_amplitude_range": float(np.ptp(filtered)),
        "detected_peak_count": int(len(peaks)),
        "mean_rr_seconds": rr_mean,
        "median_rr_seconds": rr_median,
        "rr_std_seconds": rr_std,
        "rr_rmssd_seconds": rr_rmssd,
        "estimated_rate_bpm": estimated_rate,
        "power_0_5_5": relative_power(0.5, 5),
        "power_5_15": relative_power(5, 15),
        "power_15_40": relative_power(15, 40),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing metadata: {METADATA_PATH}")

    # This is the project's own NPZ file, whose IDs may be stored as objects.
    with np.load(DATA_PATH, allow_pickle=True) as data:
        signals = data["signals"]

    metadata = pd.read_csv(METADATA_PATH).reset_index(drop=True)

    if len(signals) != len(metadata):
        raise ValueError(
            f"Row mismatch: {len(signals)} signals, "
            f"{len(metadata)} metadata rows."
        )

    required = {"record_id", "label"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(
            f"Metadata is missing columns {sorted(missing)}. "
            f"Available columns: {list(metadata.columns)}"
        )

    labels = metadata["label"].astype(str).str.strip().str.upper()
    if not set(labels).issubset({"AFIB", "NON_AFIB"}):
        raise ValueError(f"Unexpected labels: {sorted(set(labels))}")

    # Normalize record IDs so 04015 and 4015 are treated as the same record.
    def normalize_record_id(value):
        value = str(value).strip()
        if value.endswith(".0"):
            value = value[:-2]
        if value.isdigit():
            return value.lstrip("0") or "0"
        return value

    groups = metadata["record_id"].map(normalize_record_id)

    feature_rows = []
    for index, signal in enumerate(signals):
        features = extract_features(signal)
        feature_rows.append(features)

    features_df = pd.DataFrame(feature_rows)

    # Preserve exact row alignment with the clean dataset and metadata.
    feature_table = pd.concat(
        [
            metadata[["record_id", "label"]].copy(),
            features_df,
        ],
        axis=1,
    )
    feature_table.to_csv(FEATURES_PATH, index=False)

    X = features_df.replace([np.inf, -np.inf], np.nan)
    y = labels
    logo = LeaveOneGroupOut()

    predictions = []
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        logo.split(X, y, groups), start=1
    ):
        if len(set(groups.iloc[train_idx])) < 2:
            raise ValueError("Too few training recordings in a fold.")

        models = {
            "LOGISTIC": make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
            "DUMMY_MAJORITY": make_pipeline(
                SimpleImputer(strategy="median"),
                DummyClassifier(strategy="most_frequent"),
            ),
        }

        fold_predictions = {}

        for model_name, model in models.items():
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            predicted = model.predict(X.iloc[test_idx])
            fold_predictions[model_name] = predicted

            for local_position, row_index in enumerate(test_idx):
                predictions.append({
                    "model": model_name,
                    "fold": fold,
                    "record_id": metadata.iloc[row_index]["record_id"],
                    "normalized_record_id": groups.iloc[row_index],
                    "true_label": y.iloc[row_index],
                    "predicted_label": predicted[local_position],
                    "correct": (
                        y.iloc[row_index] == predicted[local_position]
                    ),
                })

        for model_name, predicted in fold_predictions.items():
            fold_results.append({
                "fold": fold,
                "held_out_record": groups.iloc[test_idx[0]],
                "model": model_name,
                "test_segments": len(test_idx),
                "train_recordings": len(set(groups.iloc[train_idx])),
                "accuracy": accuracy_score(y.iloc[test_idx], predicted),
                "balanced_accuracy": balanced_accuracy_score(
                    y.iloc[test_idx], predicted
                ),
            })

    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(PREDICTIONS_PATH, index=False)

    fold_df = pd.DataFrame(fold_results)

    report_lines = [
        "GROUPED MODEL COMPARISON",
        "=" * 65,
        f"Segments: {len(metadata)}",
        f"Unique recordings: {groups.nunique()}",
        f"Class counts:\n{y.value_counts().to_string()}",
        "",
        "All out-of-fold predictions are made with one recording held out.",
        "The same folds are used for both models.",
        "",
    ]

    for model_name in ("LOGISTIC", "DUMMY_MAJORITY"):
        subset = pred_df[pred_df["model"] == model_name]
        truth = subset["true_label"]
        predicted = subset["predicted_label"]

        report_lines.extend([
            f"MODEL: {model_name}",
            "-" * 65,
            f"Accuracy: {accuracy_score(truth, predicted):.3f}",
            f"Balanced accuracy: {balanced_accuracy_score(truth, predicted):.3f}",
            "Confusion matrix (rows=true, columns=predicted; "
            "AFIB then NON_AFIB):",
            str(confusion_matrix(
                truth,
                predicted,
                labels=["AFIB", "NON_AFIB"],
            )),
            classification_report(
                truth,
                predicted,
                labels=["AFIB", "NON_AFIB"],
                zero_division=0,
                digits=3,
            ),
            "",
        ])

    report_lines.extend([
        "PER-FOLD RESULTS",
        "-" * 65,
        fold_df.to_string(index=False, float_format=lambda value: f"{value:.3f}"),
        "",
        "CAUTIONS",
        "-" * 65,
        "Exploratory prototype only; not a clinical diagnostic model.",
        "Peak-count and RR-derived features use an unvalidated detector.",
        "Leave-one-recording-out evaluation does not guarantee patient-level "
        "independence.",
        "The dummy comparison is a reference baseline, not proof of clinical value.",
    ])

    report = "\n".join(report_lines)
    SUMMARY_PATH.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nSaved features: {FEATURES_PATH}")
    print(f"Saved predictions: {PREDICTIONS_PATH}")
    print(f"Saved summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
