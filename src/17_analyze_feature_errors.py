
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks, welch

# --------------------------------------------------
# STEP 17: Feature error analysis for grouped baseline
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

DATASET_FILE = RESULTS / "expanded_ecg_dataset_clean.npz"
PREDICTIONS_FILE = RESULTS / "grouped_baseline_predictions.csv"

FEATURE_OUTPUT = RESULTS / "grouped_baseline_features.csv"
COMPARISON_OUTPUT = RESULTS / "grouped_feature_error_comparison.csv"
REPORT_OUTPUT = RESULTS / "grouped_feature_error_report.txt"

FS = 250

FEATURE_COLUMNS = [
    "signal_std",
    "signal_rms",
    "amplitude_range",
    "robust_range",
    "iqr",
    "filtered_std",
    "peak_count",
    "rr_mean",
    "rr_std",
    "rr_median",
    "rr_rmssd",
    "estimated_rate",
    "power_0_5_5",
    "power_5_15",
    "power_15_40",
]


def standardize_record_id(value):
    value = str(value).strip()
    if value.endswith(".0"):
        value = value[:-2]
    return value.zfill(5) if value.isdigit() else value


def extract_features(signal):
    """Reproduce the feature extractor used in Step 20."""
    signal = np.asarray(signal, dtype=float)

    if not np.isfinite(signal).all():
        raise ValueError("Signal contains NaN or infinite values.")

    centered = signal - np.median(signal)
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

    peak_signal = np.abs(filtered)
    prominence = max(float(np.std(peak_signal) * 0.5), 1e-8)

    peaks, _ = find_peaks(
        peak_signal,
        distance=int(0.25 * FS),
        prominence=prominence,
    )
    features["peak_count"] = float(len(peaks))

    rr = np.array([], dtype=float)

    if len(peaks) >= 2:
        rr = np.diff(peaks) / FS
        rr = rr[(rr >= 0.25) & (rr <= 3.0)]

    if len(rr):
        features["rr_mean"] = float(np.mean(rr))
        features["rr_std"] = float(np.std(rr))
        features["rr_median"] = float(np.median(rr))
        features["rr_rmssd"] = (
            float(np.sqrt(np.mean(np.diff(rr) ** 2)))
            if len(rr) >= 2 else np.nan
        )
        features["estimated_rate"] = float(60 / np.mean(rr))
    else:
        features["rr_mean"] = np.nan
        features["rr_std"] = np.nan
        features["rr_median"] = np.nan
        features["rr_rmssd"] = np.nan
        features["estimated_rate"] = np.nan

    freqs, power = welch(
        filtered,
        fs=FS,
        nperseg=min(len(filtered), FS * 4),
    )

    # Match Step 20's spectral feature calculations.
    total_power = (
        np.trapezoid(power, freqs) if len(freqs) > 1 else 0.0
    )

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
        features[name] = (
            float(band_power / total_power)
            if total_power > 0 else np.nan
        )

    return features


def main():
    if not DATASET_FILE.exists():
        raise FileNotFoundError(DATASET_FILE)

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(PREDICTIONS_FILE)

    with np.load(DATASET_FILE, allow_pickle=True) as data:
        signals = data["signals"]
        labels = data["labels"].astype(str)
        record_ids = np.array([
            standardize_record_id(x) for x in data["record_ids"]
        ])

    predictions = pd.read_csv(PREDICTIONS_FILE)

    required = {
        "record_id",
        "true_label",
        "predicted_label",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(
            f"Prediction file is missing columns: {sorted(missing)}"
        )

    if not (len(signals) == len(labels) == len(record_ids)):
        raise ValueError("Dataset arrays have inconsistent lengths.")

    if len(predictions) != len(signals):
        raise ValueError(
            "Dataset and grouped prediction row counts differ. "
            "Check whether both files come from the same dataset version."
        )

    prediction_records = predictions["record_id"].map(
        standardize_record_id
    ).to_numpy()

    # Step 20 saves predictions in dataset order. Verify that assumption
    # before attaching features to prediction rows.
    if not np.array_equal(record_ids, prediction_records):
        raise ValueError(
            "Recording IDs are not aligned row-by-row. "
            "Do not compare features until the rows are correctly matched."
        )

    if not np.array_equal(labels, predictions["true_label"].astype(str).to_numpy()):
        raise ValueError(
            "True labels are not aligned with the saved dataset."
        )

    print("Extracting Step 20 features...")
    feature_rows = [
        extract_features(signal) for signal in signals
    ]

    features = pd.DataFrame(feature_rows)
    features.insert(0, "segment_index", np.arange(len(signals)))
    features.insert(1, "record_id", record_ids)
    features.insert(2, "true_label", labels)

    features.to_csv(FEATURE_OUTPUT, index=False)

    df = features.copy()
    df["predicted_label"] = predictions["predicted_label"].astype(str)
    df["evaluated"] = df["predicted_label"] != "NOT_EVALUATED"
    df["correct"] = df["true_label"] == df["predicted_label"]

    evaluated = df[df["evaluated"]].copy()

    if evaluated.empty:
        raise RuntimeError("There are no evaluated predictions.")

    evaluated.to_csv(COMPARISON_OUTPUT, index=False)

    report = [
        "=" * 68,
        "GROUPED BASELINE FEATURE ERROR ANALYSIS",
        "=" * 68,
        f"Segments in dataset: {len(df)}",
        f"Segments evaluated: {len(evaluated)}",
        f"Correct predictions: {int(evaluated['correct'].sum())}",
        f"Misclassified predictions: {int((~evaluated['correct']).sum())}",
        "",
        "CORRECT VS MISCLASSIFIED FEATURE SUMMARIES",
        "Descriptive comparisons only; not causal explanations.",
    ]

    for column in FEATURE_COLUMNS:
        summary = evaluated.groupby("correct")[column].agg(
            ["count", "mean", "median"]
        )
        summary.index = [
            "Misclassified" if not value else "Correct"
            for value in summary.index
        ]

        report.extend([
            "",
            f"Feature: {column}",
            summary.to_string(),
        ])

    report.extend([
        "",
        "MISCLASSIFICATION COUNTS BY RECORDING",
    ])

    by_record = evaluated.groupby("record_id").agg(
        segments=("correct", "size"),
        correct=("correct", "sum"),
        misclassified=("correct", lambda s: int((~s).sum())),
    )
    by_record["accuracy"] = by_record["correct"] / by_record["segments"]
    report.append(by_record.to_string())

    report.extend([
        "",
        "LIMITATIONS",
        "- This is an exploratory analysis of a small dataset.",
        "- Feature averages do not prove why a segment was misclassified.",
        "- Multiple segments from one recording are not independent patients.",
        "- Peak-derived timing features are approximate, not clinical measurements.",
        "- This analysis does not validate clinical performance.",
    ])

    report_text = "\n".join(report)
    REPORT_OUTPUT.write_text(report_text, encoding="utf-8")

    print("\n" + report_text)
    print(f"\nSaved features: {FEATURE_OUTPUT}")
    print(f"Saved comparison: {COMPARISON_OUTPUT}")
    print(f"Saved report: {REPORT_OUTPUT}")


if __name__ == "__main__":
    main()
