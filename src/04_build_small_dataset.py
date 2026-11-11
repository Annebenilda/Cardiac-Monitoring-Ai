
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

RECORD = "04015"
FS = 250
DURATION = 10
SAMPLES = FS * DURATION

# These start times fall inside the annotated rhythm intervals
# you inspected earlier.
segment_starts = [
    (100, "NON_AFIB"),
    (200, "NON_AFIB"),
    (420, "AFIB"),
    (540, "AFIB"),
    (700, "NON_AFIB"),
]

Path("results").mkdir(exist_ok=True)

signals = []
labels = []
metadata = []

for start_seconds, expected_label in segment_starts:
    print(f"Loading segment at {start_seconds} seconds...")

    start_sample = start_seconds * FS

    record = wfdb.rdrecord(
        RECORD,
        pn_dir="afdb",
        sampfrom=start_sample,
        sampto=start_sample + SAMPLES,
        channels=[0],
    )

    signal = record.p_signal[:, 0]

    if len(signal) != SAMPLES:
        print("Skipping segment: unexpected sample count.")
        continue

    if not np.isfinite(signal).all():
        print("Skipping segment: signal contains invalid values.")
        continue

    signals.append(signal)
    labels.append(expected_label)
    metadata.append({
        "record": RECORD,
        "start_seconds": start_seconds,
        "label": expected_label,
    })

if not signals:
    raise RuntimeError("No segments were loaded successfully.")

np.savez_compressed(
    "results/small_ecg_dataset.npz",
    signals=np.asarray(signals),
    labels=np.asarray(labels),
)

pd.DataFrame(metadata).to_csv(
    "results/small_ecg_metadata.csv",
    index=False,
)

print("\nDataset created successfully!")
print("Signal array shape:", np.asarray(signals).shape)
print("\nLabel counts:")
print(pd.Series(labels).value_counts())
print("\nFiles saved:")
print("results/small_ecg_dataset.npz")
print("results/small_ecg_metadata.csv")