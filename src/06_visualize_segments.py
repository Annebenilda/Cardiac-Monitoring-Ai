
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = np.load("results/small_ecg_dataset.npz")
signals = data["signals"]
labels = data["labels"]

fs = 250
time = np.arange(signals.shape[1]) / fs

fig, axes = plt.subplots(
    len(signals), 1,
    figsize=(12, 12),
    sharex=True
)

for i, ax in enumerate(axes):
    ax.plot(time, signals[i], linewidth=0.8)
    ax.set_ylabel("Amplitude (mV)")
    ax.set_title(f"Segment {i + 1}: {labels[i]}")
    ax.grid(True)

axes[-1].set_xlabel("Time within segment (seconds)")
fig.tight_layout()

output = Path("results/all_segments.png")
fig.savefig(output, dpi=150)
plt.close(fig)

print("Saved plot:", output.resolve())