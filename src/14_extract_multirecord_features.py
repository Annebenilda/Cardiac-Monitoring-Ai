
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, find_peaks

# Configuration
FS = 250
LOWCUT = 0.5
HIGHCUT = 40.0

DATA_PATH = Path("results/multirecord_ecg_dataset.npz")
QUALITY_PATH = Path("results/multirecord_signal_quality.csv")
OUTPUT_PATH = Path("results/multirecord_ecg_features.csv")


def extract_features(signal):
    """Extract basic signal and rhythm features from a 10-second ECG."""

    signal = np.asarray(signal, dtype=float)

    if not np.isfinite(signal).all():
        raise ValueError("Signal contains NaN or infinite values.")

    # Remove baseline drift and high-frequency noise.
    sos = butter(
        4,
        [LOWCUT, HIGHCUT],
        btype="bandpass",
        fs=FS,
        output="sos",
    )

    filtered = sosfiltfilt(sos, signal)

    # Basic signal features.
    amplitude_range = float(np.ptp(filtered))
    signal_std = float(np.std(filtered))
    rms = float(np.sqrt(np.mean(filtered ** 2)))

    # Estimate beat locations using both polarities.
    # This is a simple exploratory detector, not a validated QRS detector.
    minimum_distance = int(0.25 * FS)
    prominence = max(0.15 * signal_std, 1e-6)

    positive_peaks, _ = find_peaks(
        filtered,
        distance=minimum_distance,
        prominence=prominence,
    )

    negative_peaks, _ = find_peaks(
        -filtered,
        distance=minimum_distance,
        prominence=prominence,
    )

    peaks = (
        positive_peaks
        if len(positive_peaks) >= len(negative_peaks)
        else negative_peaks
    )

    # Estimate intervals between detected peaks.
    rr_intervals = np.diff(peaks) / FS

    if len(rr_intervals) > 0:
        mean_rr = float(np.mean(rr_intervals))
        rr_std = float(np.std(rr_intervals))
        rr_rmssd = float(
            np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
        ) if len(rr_intervals) > 1 else 0.0

        estimated_rate = (
            float(60.0 / mean_rr) if mean_rr > 0 else np.nan
        )
    else:
        mean_rr = np.nan
        rr_std = np.nan
        rr_rmssd = np.nan
        estimated_rate = np.nan

    return {
        "filtered_amplitude_range": amplitude_range,
        "filtered_signal_std": signal_std,
        "filtered_rms": rms,
        "detected_peak_count": int(len(peaks)),
        "mean_rr_seconds": mean_rr,
        "rr_std_seconds": rr_std,
        "rr_rmssd_seconds": rr_rmssd,
        "estimated_rate_bpm": estimated_rate,
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    data = np.load(DATA_PATH, allow_pickle=True)
    signals = data["signals"]
    labels = data["labels"]
    record_ids = data["record_ids"]

    if not (len(signals) == len(labels) == len(record_ids)):
        raise ValueError("Dataset arrays have inconsistent lengths.")

    # Keep quality information attached to each segment.
    quality = None
    if QUALITY_PATH.exists():
        quality = pd.read_csv(QUALITY_PATH)
        quality["record_id"] = quality["record_id"].astype(str)

    rows = []

    for i, signal in enumerate(signals):
        record_id = str(record_ids[i])
        label = str(labels[i])

        try:
            features = extract_features(signal)

            row = {
                "record_id": record_id,
                "label": label,
                "segment_index": i + 1,
                **features,
            }

            if quality is not None:
                match = quality[
                    (quality["record_id"] == record_id)
                    & (quality["label"] == label)
                ]

                # Each record should have one segment per label.
                if len(match) == 1:
                    row["quality_status"] = (
                        match.iloc[0]["quality_status"]
                    )
                    row["quality_flags"] = (
                        match.iloc[0]["quality_flags"]
                    )
                else:
                    row["quality_status"] = "NOT_MATCHED"
                    row["quality_flags"] = ""

            rows.append(row)

            print(
                f"Processed segment {i + 1:02d}: "
                f"record={record_id}, label={label}"
            )

        except Exception as exc:
            print(
                f"Skipping segment {i + 1} "
                f"(record {record_id}): {exc}"
            )

    if not rows:
        raise RuntimeError("No features were extracted.")

    df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Extracted ECG features ===")
    print(df.to_string(index=False))

    print("\nFeature dataset shape:", df.shape)
    print("\nLabel counts:")
    print(df["label"].value_counts())

    print(f"\nSaved features to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()