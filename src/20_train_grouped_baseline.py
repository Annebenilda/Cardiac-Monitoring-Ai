
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks, welch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# --------------------------------------------------
# STEP 20: Grouped baseline evaluation
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

DATASET_FILE = RESULTS / "expanded_ecg_dataset_clean.npz"
METADATA_FILE = RESULTS / "expanded_ecg_metadata_clean.csv"

PREDICTIONS_FILE = RESULTS / "grouped_baseline_predictions.csv"
METRICS_FILE = RESULTS / "grouped_baseline_metrics.txt"

FS = 250
LABELS = ["AFIB", "NON_AFIB"]


def standardize_record_id(value):
    value = str(value).strip()
    if value.endswith(".0"):
        value = value[:-2]
    return value.zfill(5) if value.isdigit() else value


def extract_features(signal):
    """Extract simple exploratory features from one ECG segment."""
    signal = np.asarray(signal, dtype=float)

    if not np.isfinite(signal).all():
        raise ValueError("Signal contains NaN or infinite values.")

    # Remove DC offset.
    centered = signal - np.median(signal)

    # Robust signal scale and amplitude features.
    q05, q25, q75, q95 = np.percentile(
        centered, [5, 25, 75, 95]
    )

    features = {
        "signal_std": float(np.std(centered)),
        "signal_rms": float(np.sqrt(np.mean(centered ** 2))),
        "amplitude_range": float(np.ptp(centered)),
        "robust_range": float(q95 - q05),
        "iqr": float(q75 - q25),
    }

    # Bandpass filtering for exploratory peak and spectral features.
    try:
        sos = butter(
            3, [0.5, 40],
            btype="bandpass",
            fs=FS,
            output="sos",
        )
        filtered = sosfiltfilt(sos, centered)
    except ValueError:
        filtered = centered.copy()

    features["filtered_std"] = float(np.std(filtered))

    # Approximate beat timing using prominent peaks in absolute signal.
    # This is not a clinical QRS detector.
    peak_signal = np.abs(filtered)
    prominence = max(float(np.std(peak_signal) * 0.5), 1e-8)

    peaks, _ = find_peaks(
        peak_signal,
        distance=int(0.25 * FS),
        prominence=prominence,
    )

    features["peak_count"] = float(len(peaks))

    if len(peaks) >= 2:
        rr = np.diff(peaks) / FS
        rr = rr[(rr >= 0.25) & (rr <= 3.0)]

        if len(rr):
            features["rr_mean"] = float(np.mean(rr))
            features["rr_std"] = float(np.std(rr))
            features["rr_median"] = float(np.median(rr))
            features["rr_rmssd"] = float(
                np.sqrt(np.mean(np.diff(rr) ** 2))
            ) if len(rr) >= 2 else np.nan
            features["estimated_rate"] = float(60 / np.mean(rr))
        else:
            features["rr_mean"] = np.nan
            features["rr_std"] = np.nan
            features["rr_median"] = np.nan
            features["rr_rmssd"] = np.nan
            features["estimated_rate"] = np.nan
    else:
        features["rr_mean"] = np.nan
        features["rr_std"] = np.nan
        features["rr_median"] = np.nan
        features["rr_rmssd"] = np.nan
        features["estimated_rate"] = np.nan

    # Relative power in broad frequency bands.
    freqs, power = welch(filtered, fs=FS, nperseg=min(len(filtered), FS * 4))

    total_power = np.trapezoid(power, freqs) if len(freqs) > 1 else 0.0

    bands = {
        "power_0_5_5": (0.5, 5.0),
        "power_5_15": (5.0, 15.0),
        "power_15_40": (15.0, 40.0),
    }

    for name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)

        band_power = (
            np.trapezoid(power[mask], freqs[mask])
            if np.count_nonzero(mask) >= 2
            else 0.0
        )

        features[name] = float(
            band_power / total_power
        ) if total_power > 0 else np.nan

    return features


def main():
    if not DATASET_FILE.exists() or not METADATA_FILE.exists():
        raise FileNotFoundError(
            "Cleaned dataset files were not found. "
            "Run Step 19B first."
        )

   
with np.load(DATASET_FILE, allow_pickle=True) as data:
    signals = data["signals"]
    labels = data["labels"].astype(str)
    record_ids = data["record_ids"].astype(str)

    metadata = pd.read_csv(METADATA_FILE)

    if not (len(signals) == len(labels) == len(record_ids)
            == len(metadata)):
        raise ValueError("Dataset and metadata lengths do not match.")

    groups = np.array([
        standardize_record_id(x) for x in record_ids
    ])

    metadata_groups = metadata["record_id"].map(
        standardize_record_id
    ).to_numpy()

    metadata_labels = metadata["label"].astype(str).to_numpy()

    if not np.array_equal(groups, metadata_groups):
        raise ValueError("Recording IDs are not aligned.")

    if not np.array_equal(labels, metadata_labels):
        raise ValueError("Labels are not aligned.")

    if not np.isfinite(signals).all():
        raise ValueError("Non-finite signal values found.")

    unknown = set(labels) - set(LABELS)
    if unknown:
        raise ValueError(f"Unexpected labels: {unknown}")

    print("=" * 60)
    print("GROUPED ECG BASELINE EVALUATION")
    print("=" * 60)
    print(f"Segments: {len(signals)}")
    print(f"Unique recordings: {len(np.unique(groups))}")
    print("\nClass counts:")
    print(pd.Series(labels).value_counts().to_string())

    print("\nExtracting features...")
    feature_rows = []

    for i, signal in enumerate(signals):
        row = extract_features(signal)
        row["segment_index"] = i
        feature_rows.append(row)

    feature_df = pd.DataFrame(feature_rows)
    feature_columns = [
        c for c in feature_df.columns if c != "segment_index"
    ]

    X = feature_df[feature_columns].to_numpy(dtype=float)
    y = labels

    # Leave one complete recording out per test fold.
    logo = LeaveOneGroupOut()
    predictions = np.empty(len(y), dtype=object)
    probabilities = np.full(len(y), np.nan, dtype=float)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        logo.split(X, y, groups), start=1
    ):
        train_classes = set(y[train_idx])

        if len(train_classes) < 2:
            warnings.warn(
                f"Fold {fold} skipped: training data has only "
                f"one class."
            )
            predictions[test_idx] = "NOT_EVALUATED"
            continue

        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        )

        model.fit(X[train_idx], y[train_idx])
        predictions[test_idx] = model.predict(X[test_idx])

        class_list = list(model.named_steps[
            "logisticregression"
        ].classes_)

        if "AFIB" in class_list:
            afib_col = class_list.index("AFIB")
            probabilities[test_idx] = model.predict_proba(
                X[test_idx]
            )[:, afib_col]

        fold_results.append({
            "fold": fold,
            "test_recording": groups[test_idx][0],
            "test_segments": len(test_idx),
            "train_recordings": len(np.unique(groups[train_idx])),
        })

    evaluated = predictions != "NOT_EVALUATED"

    if not np.any(evaluated):
        raise RuntimeError("No folds were successfully evaluated.")

    y_true = y[evaluated]
    y_pred = predictions[evaluated]

    accuracy = accuracy_score(y_true, y_pred)
    balanced = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, pos_label="AFIB", zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, pos_label="AFIB", zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, pos_label="AFIB", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    output = pd.DataFrame({
        "record_id": groups,
        "true_label": y,
        "predicted_label": predictions,
        "estimated_afib_probability": probabilities,
    })

    output.to_csv(PREDICTIONS_FILE, index=False)

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
    )

    summary = [
        "=" * 60,
        "GROUPED BASELINE RESULTS",
        "=" * 60,
        f"Segments evaluated: {int(evaluated.sum())}/{len(y)}",
        f"Recordings: {len(np.unique(groups))}",
        f"Accuracy: {accuracy:.3f}",
        f"Balanced accuracy: {balanced:.3f}",
        f"AFIB precision: {precision:.3f}",
        f"AFIB recall: {recall:.3f}",
        f"AFIB F1 score: {f1:.3f}",
        "",
        "Confusion matrix (rows=true, columns=predicted)",
        f"Label order: {LABELS}",
        str(cm),
        "",
        "Classification report:",
        report,
        "",
        "Fold summary:",
        pd.DataFrame(fold_results).to_string(index=False),
        "",
        "Important: exploratory research prototype only. "
        "Not a clinical diagnostic model.",
    ]

    report_text = "\n".join(summary)
    print("\n" + report_text)

    METRICS_FILE.write_text(report_text, encoding="utf-8")

    print(f"\nPredictions saved to: {PREDICTIONS_FILE}")
    print(f"Metrics saved to: {METRICS_FILE}")


if __name__ == "__main__":
    main()
