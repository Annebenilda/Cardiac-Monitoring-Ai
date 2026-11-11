
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, find_peaks, welch

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

DATASET_FILE = RESULTS / "expanded_ecg_dataset_clean.npz"
METADATA_FILE = RESULTS / "expanded_ecg_metadata_clean.csv"

REPORT_FILE = RESULTS / "feature_comparison_report.csv"
PLOT_FILE = RESULTS / "feature_comparison.png"

FS = 250


def extract_features(signal):
    signal = np.asarray(signal, dtype=float)
    centered = signal - np.median(signal)

    q05, q25, q75, q95 = np.percentile(
        centered, [5, 25, 75, 95]
    )

    features = {
        "signal_std": np.std(centered),
        "signal_rms": np.sqrt(np.mean(centered ** 2)),
        "amplitude_range": np.ptp(centered),
        "robust_range": q95 - q05,
        "iqr": q75 - q25,
    }

    sos = butter(
        3, [0.5, 40],
        btype="bandpass",
        fs=FS,
        output="sos",
    )
    filtered = sosfiltfilt(sos, centered)
    features["filtered_std"] = np.std(filtered)

    prominence = max(float(np.std(np.abs(filtered)) * 0.5), 1e-8)
    peaks, _ = find_peaks(
        np.abs(filtered),
        distance=int(0.25 * FS),
        prominence=prominence,
    )
    features["peak_count"] = len(peaks)

    if len(peaks) >= 2:
        rr = np.diff(peaks) / FS
        rr = rr[(rr >= 0.25) & (rr <= 3.0)]
    else:
        rr = np.array([])

    features["rr_mean"] = np.mean(rr) if len(rr) else np.nan
    features["rr_std"] = np.std(rr) if len(rr) else np.nan
    features["rr_median"] = np.median(rr) if len(rr) else np.nan

    freqs, power = welch(
        filtered, fs=FS, nperseg=min(len(filtered), FS * 4)
    )
    total_power = np.trapezoid(power, freqs)

    for name, low, high in [
        ("power_0_5_5", 0.5, 5),
        ("power_5_15", 5, 15),
        ("power_15_40", 15, 40),
    ]:
        mask = (freqs >= low) & (freqs < high)
        band_power = (
            np.trapezoid(power[mask], freqs[mask])
            if mask.sum() >= 2 else 0.0
        )
        features[name] = (
            band_power / total_power if total_power > 0 else np.nan
        )

    return features


def main():
    with np.load(DATASET_FILE, allow_pickle=True) as data:
        signals = data["signals"]
        labels = data["labels"].astype(str)

    metadata = pd.read_csv(METADATA_FILE)

    if len(signals) != len(metadata) or len(labels) != len(metadata):
        raise ValueError("Dataset and metadata row counts differ.")

    if not np.array_equal(labels, metadata["label"].astype(str).to_numpy()):
        raise ValueError("Dataset labels and metadata labels differ.")

    print("Extracting features from cleaned ECG segments...")
    features = pd.DataFrame(
        [extract_features(signal) for signal in signals]
    )
    features["label"] = labels

    rows = []

    for column in features.columns:
        if column == "label":
            continue

        afib = features.loc[features["label"] == "AFIB", column].dropna()
        non_afib = features.loc[
            features["label"] == "NON_AFIB", column
        ].dropna()

        if len(afib) == 0 or len(non_afib) == 0:
            continue

        afib_mean = afib.mean()
        non_mean = non_afib.mean()
        pooled_scale = np.sqrt(
            (afib.var(ddof=1) + non_afib.var(ddof=1)) / 2
        )

        standardized_difference = (
            (afib_mean - non_mean) / pooled_scale
            if pooled_scale > 0 else np.nan
        )

        rows.append({
            "feature": column,
            "AFIB_mean": afib_mean,
            "AFIB_std": afib.std(),
            "NON_AFIB_mean": non_mean,
            "NON_AFIB_std": non_afib.std(),
            "mean_difference_AFIB_minus_NON_AFIB": afib_mean - non_mean,
            "standardized_difference_exploratory": standardized_difference,
            "AFIB_valid_segments": len(afib),
            "NON_AFIB_valid_segments": len(non_afib),
        })

    report = pd.DataFrame(rows)
    report["absolute_standardized_difference"] = (
        report["standardized_difference_exploratory"].abs()
    )
    report = report.sort_values(
        "absolute_standardized_difference",
        ascending=False,
    )

    report.to_csv(REPORT_FILE, index=False)

    print("\nFeature comparison (largest standardized differences first):")
    print(report.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Plot features using z-scored values for visual comparison.
    feature_names = report["feature"].tolist()
    means = []
    for name in feature_names:
        values = features[name]
        scale = values.std()
        if pd.isna(scale) or scale == 0:
            means.append((0.0, 0.0))
        else:
            means.append((
                (features.loc[features["label"] == "AFIB", name].mean()
                 - values.mean()) / scale,
                (features.loc[features["label"] == "NON_AFIB", name].mean()
                 - values.mean()) / scale,
            ))

    y = np.arange(len(feature_names))
    height = 0.36

    fig, ax = plt.subplots(figsize=(10, max(5, len(feature_names) * 0.45)))
    ax.barh(y - height / 2, [m[0] for m in means], height, label="AFIB")
    ax.barh(y + height / 2, [m[1] for m in means], height, label="NON_AFIB")
    ax.set_yticks(y)
    ax.set_yticklabels(feature_names)
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("Feature mean relative to overall mean (standard deviations)")
    ax.set_title("Exploratory ECG Feature Comparison")
    ax.legend()
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=160)
    plt.close(fig)

    print(f"\nCSV report saved to: {REPORT_FILE}")
    print(f"Feature plot saved to: {PLOT_FILE}")
    print("\nThese are descriptive comparisons, not proof of clinical significance.")


if __name__ == "__main__":
    main()
