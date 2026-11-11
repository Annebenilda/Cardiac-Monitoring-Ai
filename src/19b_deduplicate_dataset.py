
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

DATASET_FILE = RESULTS / "expanded_ecg_dataset.npz"
METADATA_FILE = RESULTS / "expanded_ecg_metadata.csv"

OUT_DATASET = RESULTS / "expanded_ecg_dataset_clean.npz"
OUT_METADATA = RESULTS / "expanded_ecg_metadata_clean.csv"


def standardize_record_id(value):
    value = str(value).strip()
    if value.endswith(".0"):
        value = value[:-2]
    return value.zfill(5) if value.isdigit() else value


print("=" * 60)
print("SAFE ECG DUPLICATE CHECK")
print("=" * 60)

with np.load(DATASET_FILE, allow_pickle=False) as data:
    signals = data["signals"]
    labels = data["labels"].astype(str)
    record_ids = data["record_ids"].astype(str)

metadata = pd.read_csv(METADATA_FILE)

if not (len(signals) == len(labels) == len(record_ids) == len(metadata)):
    raise ValueError("Dataset arrays and metadata lengths do not match.")

required = {"record_id", "start_seconds", "label"}
if not required.issubset(metadata.columns):
    raise ValueError(f"Metadata must contain: {required}")

metadata = metadata.copy().reset_index(drop=True)
metadata["record_id"] = metadata["record_id"].map(standardize_record_id)
metadata["label"] = metadata["label"].astype(str).str.strip()
metadata["start_seconds"] = pd.to_numeric(
    metadata["start_seconds"], errors="raise"
)

array_ids = pd.Series(record_ids).map(standardize_record_id)

if not np.array_equal(array_ids.to_numpy(), metadata["record_id"].to_numpy()):
    raise ValueError("Recording IDs do not align between arrays and metadata.")

if not np.array_equal(labels, metadata["label"].to_numpy()):
    raise ValueError("Labels do not align between arrays and metadata.")

if not np.isfinite(signals).all():
    raise ValueError("Non-finite signal values detected.")

# Group segments by standardized recording ID and start time.
keys = list(zip(metadata["record_id"], metadata["start_seconds"]))
groups = {}

for index, key in enumerate(keys):
    groups.setdefault(key, []).append(index)

duplicate_groups = {
    key: indices for key, indices in groups.items()
    if len(indices) > 1
}

print(f"Total segments: {len(signals)}")
print(f"Repeated recording/time groups: {len(duplicate_groups)}")
print(f"Rows in repeated groups: "
      f"{sum(len(v) for v in duplicate_groups.values())}")

# Only remove a repeated segment if its label and waveform match.
remove_indices = set()

for key, indices in duplicate_groups.items():
    first = indices[0]

    for other in indices[1:]:
        same_label = labels[first] == labels[other]
        same_signal = np.array_equal(signals[first], signals[other])

        if not same_label or not same_signal:
            print("\nCONFLICT FOUND")
            print(f"Recording/time: {key}")
            print(f"Rows: {first} and {other}")
            print(f"Labels: {labels[first]} / {labels[other]}")
            print(f"Signals exactly equal: {same_signal}")
            raise RuntimeError(
                "Conflicting repeated segments found. "
                "No cleaned files were saved; inspect this conflict first."
            )

        remove_indices.add(other)

keep_indices = [
    i for i in range(len(signals)) if i not in remove_indices
]

clean_signals = signals[keep_indices]
clean_labels = labels[keep_indices]
clean_record_ids = array_ids.to_numpy()[keep_indices]
clean_metadata = metadata.iloc[keep_indices].copy().reset_index(drop=True)

# Save to new files; preserve the original dataset.
np.savez_compressed(
    OUT_DATASET,
    signals=clean_signals,
    labels=clean_labels,
    record_ids=clean_record_ids,
)

clean_metadata.to_csv(OUT_METADATA, index=False)

print("\nSAFE DEDUPLICATION COMPLETE")
print(f"Segments before: {len(signals)}")
print(f"Segments removed: {len(remove_indices)}")
print(f"Segments after: {len(clean_signals)}")
print(f"Unique recordings: {clean_metadata['record_id'].nunique()}")
print("\nClean label counts:")
print(clean_metadata["label"].value_counts().to_string())
print(f"\nClean dataset: {OUT_DATASET}")
print(f"Clean metadata: {OUT_METADATA}")
print("\nOriginal files were not modified.")
