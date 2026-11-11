
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

data = np.load("results/small_ecg_dataset.npz")
signals = data["signals"]
labels = data["labels"]

fs = 250
time = np.arange(signals.shape[1]) / fs

# Compare one ordinary-looking segment and the flagged segment.
segments_to_check = [0, 4]

sos = butter(
    3,
    [0.5, 40],
    btype="bandpass",
    fs=fs,
    output="sos"
)

fig, axes = plt.subplots(
    len(segments_to_check), 1,
    figsize=(12, 8)
)

for ax, idx in zip(axes, segments_to_check):
    raw = signals[idx]
    filtered = sosfiltfilt(sos, raw)

    ax.plot(time, raw, label="Raw ECG", alpha=0.7)
    ax.plot(time, filtered, label="Filtered ECG", linewidth=1)
    ax.set_title(f"Segment {idx + 1}: {labels[idx]}")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude (mV)")
    ax.legend()
    ax.grid(True)

fig.tight_layout()

Path("results").mkdir(exist_ok=True)
fig.savefig("results/raw_vs_filtered.png", dpi=150)
plt.close(fig)

print("Saved comparison to results/raw_vs_filtered.png")
print("Open the image and compare the waveform shapes.")