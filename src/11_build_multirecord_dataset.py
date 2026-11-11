
import wfdb
import numpy as np
import pandas as pd
from pathlib import Path

FS = 250
DURATION = 10
SAMPLES = FS * DURATION
RECORDS = ["04015", "04043", "04048", "04126", "04746"]

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

segments = []
metadata = []

for record_id in RECORDS:
    print(f"\nChecking record {record_id}...")

    try:
        ann = wfdb.rdann(record_id, "atr", pn_dir="afdb")
        changes = sorted(zip(ann.sample, ann.aux_note))

        selected_labels = set()

        for i, (start, note) in enumerate(changes):
            label = note.strip()

            if label not in ("(AFIB", "(N"):
                continue

            # One example of each label per record.
            class_name = "AFIB" if label == "(AFIB" else "NON_AFIB"
            if class_name in selected_labels:
                continue

            end = (
                changes[i + 1][0]
                if i + 1 < len(changes)
                else float("inf")
            )

            # Start one second after the annotation boundary.
            segment_start = int(start) + FS
            segment_end = segment_start + SAMPLES

            if segment_end > end:
                continue

            try:
                rec = wfdb.rdrecord(
                    record_id,
                    pn_dir="afdb",
                    sampfrom=segment_start,
                    sampto=segment_end,
                    channels=[0],
                )

                signal = rec.p_signal[:, 0]

                if len(signal) != SAMPLES:
                    print("  Skipping incomplete segment.")
                    continue

                segments.append(signal.astype(np.float32))
                metadata.append({
                    "record_id": record_id,
                    "start_seconds": segment_start / FS,
                    "label": class_name,
                    "reference_annotation": label,
                })

                selected_labels.add(class_name)
                print(
                    f"  Added {class_name} segment "
                    f"at {segment_start / FS:.2f}s"
                )

            except Exception as exc:
                print(f"  Signal download failed: {exc}")

        if not selected_labels:
            print("  No suitable segments found.")

    except Exception as exc:
        print(f"  Could not read annotations: {exc}")

if not segments:
    raise RuntimeError(
        "No segments collected. Check internet access and record IDs."
    )

signals = np.stack(segments)
df = pd.DataFrame(metadata)

np.savez_compressed(
    OUTPUT_DIR / "multirecord_ecg_dataset.npz",
    signals=signals,
    labels=df["label"].to_numpy(),
    record_ids=df["record_id"].to_numpy(),
)

df.to_csv(
    OUTPUT_DIR / "multirecord_ecg_metadata.csv",
    index=False,
)

print("\nDataset shape:", signals.shape)
print("\nLabel counts:")
print(df["label"].value_counts())
print("\nRecords represented:")
print(df["record_id"].nunique())
print("\nSaved dataset and metadata in results/")