
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks

# Load our saved dataset
data = np.load("results/small_ecg_dataset.npz")
signals = data["signals"]
labels = data["labels"]

FS = 250
FEATURES = []

def extract_features(ecg, fs=FS):
    # Basic signal checks
    ecg = np.asarray(ecg, dtype=float)

    # Bandpass filter to reduce baseline drift and high-frequency noise
    sos = butter(
        3, [0.5, 40],
        btype="bandpass",
        fs=fs,
        output="sos"
    )
    filtered = sosfiltfilt(sos, ecg)

    # Find approximate QRS peaks for either ECG polarity
    min_distance = int(0.25 * fs)
    prominence = max(0.1 * np.std(filtered), 1e-6)

    positive, _ = find_peaks(
        filtered,
        distance=min_distance,
        prominence=prominence
    )
    negative, _ = find_peaks(
        -filtered,
        distance=min_distance,
        prominence=prominence
    )

    peaks = positive if len(positive) >= len(negative) else negative

    # Calculate intervals between detected peaks
    rr = np.diff(peaks) / fs

    features = {
        "mean_amplitude": np.mean(filtered),
        "signal_std": np.std(filtered),
        "signal_rms": np.sqrt(np.mean(filtered ** 2)),
        "peak_to_peak": np.ptp(filtered),
        "detected_peaks": len(peaks),
        "mean_rr_seconds": np.mean(rr) if len(rr) else np.nan,
        "rr_std_seconds": np.std(rr) if len(rr) >= 2 else np.nan,
        "rr_rmssd_seconds": (
            np.sqrt(np.mean(np.diff(rr) ** 2))
            if len(rr) >= 2 else np.nan
        ),
    }

    return features


for i, signal in enumerate(signals):
    features = extract_features(signal)
    features["segment_id"] = i + 1
    features["label"] = str(labels[i])
    FEATURES.append(features)

df = pd.DataFrame(FEATURES)

Path("results").mkdir(exist_ok=True)
df.to_csv("results/ecg_features.csv", index=False)

print("\nExtracted ECG features:")
print(df.round(4).to_string(index=False))

print("\nSaved to results/ecg_features.csv")