
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks, welch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------
# Train a reusable model for the interactive demo.
# This is separate from the grouped evaluation.
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"

DATA_PATH = RESULTS_DIR / "expanded_ecg_dataset_clean.npz"
METADATA_PATH = RESULTS_DIR / "expanded_ecg_metadata_clean.csv"
MODEL_PATH = RESULTS_DIR / "cardiac_demo_model.joblib"

FS = 250
LABELS = {"AFIB", "NON_AFIB"}


def extract_features(signal):
    """Keep this feature extraction consistent with the baseline."""
    signal = np.asarray(signal, dtype=float).reshape(-1)

    if len(signal) < FS * 2 or not np.isfinite(signal).all():
        raise ValueError(
            "Signal must contain at least 2 seconds of finite samples."
        )

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

    sos = butter(
        3, [0.5, 40],
        btype="bandpass",
        fs=FS,
        output="sos",
    )

    try:
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

    if len(peaks) >= 2:
        rr = np.diff(peaks) / FS
        rr = rr[(rr >= 0.25) & (rr <= 3.0)]
    else:
        rr = np.array([], dtype=float)

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

    frequencies, power = welch(
        filtered,
        fs=FS,
        nperseg=min(len(filtered), FS * 4),
    )

    total_power = (
        np.trapezoid(power, frequencies)
        if len(frequencies) > 1 else 0.0
    )

    bands = {
        "power_0_5_5": (0.5, 5.0),
        "power_5_15": (5.0, 15.0),
        "power_15_40": (15.0, 40.0),
    }

    for name, (low, high) in bands.items():
        mask = (frequencies >= low) & (frequencies < high)

        band_power = (
            np.trapezoid(power[mask], frequencies[mask])
            if np.count_nonzero(mask) >= 2 else 0.0
        )

        features[name] = (
            float(band_power / total_power)
            if total_power > 0 else np.nan
        )

    return features


def main():
    if not DATA_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Clean dataset files are missing. "
            "Check results/expanded_ecg_dataset_clean.npz "
            "and results/expanded_ecg_metadata_clean.csv."
        )

    # This is the project's own locally generated dataset.
    with np.load(DATA_PATH, allow_pickle=True) as data:
        signals = data["signals"]
        labels = data["labels"].astype(str)

    metadata = pd.read_csv(METADATA_PATH)

    if len(signals) != len(labels) or len(labels) != len(metadata):
        raise ValueError("Signals, labels, and metadata are misaligned.")

    labels = np.char.upper(np.char.strip(labels))

    if not set(labels).issubset(LABELS):
        raise ValueError(f"Unexpected labels: {set(labels) - LABELS}")

    if len(set(labels)) < 2:
        raise ValueError("Training requires both classes.")

    if not np.isfinite(signals).all():
        raise ValueError("Dataset contains non-finite signal values.")

    print(f"Extracting features from {len(signals)} segments...")

    feature_rows = [
        extract_features(signal) for signal in signals
    ]
    feature_df = pd.DataFrame(feature_rows)
    feature_columns = feature_df.columns.tolist()

    X = feature_df.replace([np.inf, -np.inf], np.nan)

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, labels)

    for warning in caught:
        print(f"Training warning: {warning.message}")

    bundle = {
        "model": model,
        "feature_columns": feature_columns,
        "sampling_frequency": FS,
        "training_segments": int(len(signals)),
        "class_labels": sorted(set(labels)),
        "purpose": "Experimental demonstration only",
    }

    joblib.dump(bundle, MODEL_PATH)

    print("\nDemo model saved successfully.")
    print(f"File: {MODEL_PATH}")
    print(f"Training segments: {len(signals)}")
    print(f"Feature count: {len(feature_columns)}")
    print(f"Classes: {sorted(set(labels))}")
    print("\nImportant: this model was fitted on the full dataset.")
    print("Its predictions on these same segments are not held-out tests.")
    print("Do not use it for diagnosis or healthcare decisions.")


if __name__ == "__main__":
    main()

