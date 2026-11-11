
import numpy as np
import pandas as pd
from pathlib import Path

data_path = Path("results/multirecord_ecg_dataset.npz")
metadata_path = Path("results/multirecord_ecg_metadata.csv")

data = np.load(data_path, allow_pickle=True)
signals = data["signals"]
labels = data["labels"]
record_ids = data["record_ids"]
metadata = pd.read_csv(metadata_path)

print("=== Dataset validation ===")
print("Signal array shape:", signals.shape)
print("Labels shape:", labels.shape)
print("Record IDs shape:", record_ids.shape)

assert signals.ndim == 2, "Signals should be a 2D array."
assert signals.shape[1] == 2500, "Each segment should contain 2500 samples."
assert len(signals) == len(labels) == len(record_ids) == len(metadata)

print("\nMissing/non-finite signal values:", 
      int((~np.isfinite(signals)).sum()))

print("\nLabel counts:")
print(pd.Series(labels).value_counts())

print("\nSegments per record:")
print(metadata.groupby("record_id")["label"].value_counts())

print("\nSignal amplitude summary:")
for i, signal in enumerate(signals):
    print(
        f"{i + 1:02d} | record={record_ids[i]} "
        f"| label={labels[i]} "
        f"| min={signal.min():.3f} "
        f"| max={signal.max():.3f} "
        f"| std={signal.std():.3f}"
    )

if not np.isfinite(signals).all():
    print("\nWARNING: Non-finite values detected.")
else:
    print("\nPASS: All signal values are finite.")

print("\nValidation complete.")