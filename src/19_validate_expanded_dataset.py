
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------
# STEP 19: Validate the expanded ECG dataset
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

DATASET_FILE = RESULTS / "expanded_ecg_dataset.npz"
METADATA_FILE = RESULTS / "expanded_ecg_metadata.csv"
CLEAN_METADATA_FILE = RESULTS / "expanded_ecg_metadata_clean.csv"

print("=" * 60)
print("EXPANDED ECG DATASET VALIDATION")
print("=" * 60)

# 1. Check that both required files exist
if not DATASET_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_FILE}")

if not METADATA_FILE.exists():
    raise FileNotFoundError(f"Metadata not found: {METADATA_FILE}")

# 2. Load the dataset
with np.load(DATASET_FILE, allow_pickle=False) as data:
    required_keys = {"signals", "labels", "record_ids"}

    if not required_keys.issubset(data.files):
        raise ValueError(
            f"Missing dataset fields. Found: {data.files}"
        )

    signals = data["signals"]
    labels = data["labels"].astype(str)
    record_ids = data["record_ids"].astype(str)

metadata = pd.read_csv(METADATA_FILE)

print("\n1. Dataset dimensions")
print(f"Signals shape: {signals.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Record IDs shape: {record_ids.shape}")
print(f"Metadata shape: {metadata.shape}")

# 3. Check signal dimensions and array alignment
if signals.ndim != 2:
    raise ValueError("Expected signals with shape (segments, samples).")

if not (len(signals) == len(labels) == len(record_ids)):
    raise ValueError(
        "Signals, labels, and record IDs have different lengths."
    )

if len(metadata) != len(signals):
    raise ValueError(
        "Metadata row count does not match the number of signals."
    )

print("\n2. Signal integrity")
print(f"All signal values finite: {np.isfinite(signals).all()}")
print(f"NaN values: {np.isnan(signals).sum()}")
print(f"Infinite values: {np.isinf(signals).sum()}")

if not np.isfinite(signals).all():
    raise ValueError("Non-finite signal values found.")

if signals.shape[1] != 2500:
    print(
        "WARNING: Expected 2500 samples per segment "
        "(10 seconds at 250 Hz)."
    )
else:
    print("Segment duration: 10 seconds at 250 Hz")

print(f"Overall minimum amplitude: {signals.min():.4f}")
print(f"Overall maximum amplitude: {signals.max():.4f}")
print(f"Median segment standard deviation: "
      f"{np.median(np.std(signals, axis=1)):.4f}")

# 4. Check labels
print("\n3. Rhythm label distribution")
print(pd.Series(labels).value_counts().to_string())

allowed_labels = {"AFIB", "NON_AFIB"}
unexpected_labels = set(labels) - allowed_labels

if unexpected_labels:
    print(f"WARNING: Unexpected labels: {unexpected_labels}")
else:
    print("All labels belong to the expected label set.")

# 5. Standardize recording IDs
# Examples: 4015 -> 04015, 4908 -> 04908
def standardize_record_id(value):
    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    if value.isdigit():
        return value.zfill(5)

    return value

normalized_ids = pd.Series(record_ids).map(
    standardize_record_id
)

print("\n4. Recording ID consistency")
print(f"Unique IDs before standardization: {len(set(record_ids))}")
print(f"Unique IDs after standardization: {normalized_ids.nunique()}")

id_mapping = pd.DataFrame({
    "original_id": record_ids,
    "standardized_id": normalized_ids
}).drop_duplicates()

print("\nID mappings:")
print(id_mapping.to_string(index=False))

# 6. Check metadata columns and consistency
print("\n5. Metadata checks")
print("Metadata columns:", list(metadata.columns))

required_columns = {"record_id", "label"}
missing_columns = required_columns - set(metadata.columns)

if missing_columns:
    raise ValueError(
        f"Metadata is missing required columns: {missing_columns}"
    )

metadata = metadata.copy()

metadata["record_id"] = metadata["record_id"].map(
    standardize_record_id
)
metadata["label"] = metadata["label"].astype(str).str.strip()

arrays_df = pd.DataFrame({
    "record_id": normalized_ids,
    "label": labels
})

arrays_df["label"] = arrays_df["label"].str.strip()

metadata_pairs = list(
    zip(metadata["record_id"], metadata["label"])
)
array_pairs = list(
    zip(arrays_df["record_id"], arrays_df["label"])
)

if metadata_pairs == array_pairs:
    print("Metadata order and labels match the dataset arrays.")
else:
    print(
        "WARNING: Metadata order or labels do not exactly match "
        "the dataset arrays."
    )

# 7. Check duplicate segment identifiers
print("\n6. Duplicate segment check")

if "start_seconds" in metadata.columns:
    metadata["start_seconds"] = pd.to_numeric(
        metadata["start_seconds"], errors="coerce"
    )

    duplicate_mask = metadata.duplicated(
        subset=["record_id", "start_seconds"],
        keep=False
    )

    print(f"Rows involved in duplicate segment IDs: "
          f"{int(duplicate_mask.sum())}")

    if duplicate_mask.any():
        print(metadata.loc[
            duplicate_mask,
            ["record_id", "start_seconds", "label"]
        ].to_string(index=False))
else:
    print("No start_seconds column; segment duplicates not checked.")

# 8. Count segments per recording and label
print("\n7. Segments per recording and label")

summary = pd.crosstab(
    metadata["record_id"],
    metadata["label"],
    margins=False
)

print(summary.to_string())

# 9. Save standardized metadata separately
metadata.to_csv(CLEAN_METADATA_FILE, index=False)

print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print(f"Cleaned metadata saved to: {CLEAN_METADATA_FILE}")
print("The original dataset files were not modified.")
print("=" * 60)
