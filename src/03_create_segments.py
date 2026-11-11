
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

RECORD_NAME = "04015"
SAMPLE_RATE = 250
SEGMENT_SECONDS = 10
SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_SECONDS

Path("results").mkdir(exist_ok=True)

print("Loading ECG signal...")
record = wfdb.rdrecord(RECORD_NAME, pn_dir="afdb")
ecg = record.p_signal[:, 0]
fs = int(record.fs)

print("Loading rhythm annotations...")
ann = wfdb.rdann(RECORD_NAME, "atr", pn_dir="afdb")

# Rhythm annotations define intervals: each label continues
# until the next rhythm annotation.
rhythm_changes = sorted(
    zip(ann.sample, ann.aux_note),
    key=lambda item: item[0]
)

segments = []
ignored = 0

for i, (start, rhythm) in enumerate(rhythm_changes):
    end = (
        rhythm_changes[i + 1][0]
        if i + 1 < len(rhythm_changes)
        else len(ecg)
    )

    rhythm = rhythm.strip()

    # Keep only explicitly labelled normal and AF intervals.
    if rhythm == "(AFIB":
        label = "AFIB"
    elif rhythm == "(N":
        label = "NON_AFIB"
    else:
        ignored += 1
        continue

    # Create non-overlapping 10-second segments within this interval.
    for seg_start in range(start, end - SEGMENT_SAMPLES + 1,
                           SEGMENT_SAMPLES):
        segment = ecg[seg_start:seg_start + SEGMENT_SAMPLES]

        if len(segment) == SEGMENT_SAMPLES and np.isfinite(segment).all():
            segments.append({
                "record": RECORD_NAME,
                "start_seconds": seg_start / fs,
                "label": label,
                "signal": segment.tolist()
            })

if not segments:
    raise RuntimeError(
        "No segments created. Check the record and annotation labels."
    )

# Save segment metadata and the waveform samples.
metadata = pd.DataFrame([
    {
        "record": row["record"],
        "start_seconds": row["start_seconds"],
        "label": row["label"]
    }
    for row in segments
])

metadata.to_csv("results/segment_metadata.csv", index=False)

np.savez_compressed(
    "results/ecg_segments.npz",
    signals=np.array([row["signal"] for row in segments]),
    labels=np.array([row["label"] for row in segments]),
    start_seconds=np.array(
        [row["start_seconds"] for row in segments]
    )
)

print("\nSegments created:", len(segments))
print("\nLabel counts:")
print(metadata["label"].value_counts())
print("\nSaved:")
print("results/segment_metadata.csv")
print("results/ecg_segments.npz")
print("\nNote: These segments come from one recording only.")