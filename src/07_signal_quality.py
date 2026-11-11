
from pathlib import Path

import numpy as np
import pandas as pd

data = np.load("results/small_ecg_dataset.npz")
signals = data["signals"]
labels = data["labels"]

rows = []

for i, ecg in enumerate(signals):
    ecg = np.asarray(ecg, dtype=float)

    amplitude_range = np.ptp(ecg)
    signal_std = np.std(ecg)
    max_step = np.max(np.abs(np.diff(ecg)))
    flat_fraction = np.mean(np.abs(np.diff(ecg)) < 1e-5)

    # These are preliminary review flags, not clinical thresholds.
    flags = []

    if amplitude_range > 6:
        flags.append("large_amplitude_range")

    if max_step > 2:
        flags.append("abrupt_change")

    if flat_fraction > 0.1:
        flags.append("possible_flatline")

    rows.append({
        "segment_id": i + 1,
        "label": str(labels[i]),
        "amplitude_range": amplitude_range,
        "signal_std": signal_std,
        "largest_sample_change": max_step,
        "flat_fraction": flat_fraction,
        "review_flags": ", ".join(flags) if flags else "none",
    })

df = pd.DataFrame(rows)
Path("results").mkdir(exist_ok=True)
df.to_csv("results/signal_quality_review.csv", index=False)

print(df.round(4).to_string(index=False))
print("\nSaved to results/signal_quality_review.csv")