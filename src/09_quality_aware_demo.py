
from pathlib import Path

import numpy as np
import pandas as pd

data = np.load("results/small_ecg_dataset.npz")
signals = data["signals"]
labels = data["labels"]

results = []

for i, ecg in enumerate(signals):
    ecg = np.asarray(ecg, dtype=float)

    amplitude_range = np.ptp(ecg)
    largest_change = np.max(np.abs(np.diff(ecg)))

    flags = []

    # Preliminary engineering rules for this demonstration only.
    if amplitude_range > 6:
        flags.append("large_amplitude_range")

    if largest_change > 2:
        flags.append("abrupt_change")

    if flags:
        decision = "INCONCLUSIVE"
        reason = ", ".join(flags)
    else:
        decision = "REVIEWABLE"
        reason = "No configured quality flag triggered"

    results.append({
        "segment_id": i + 1,
        "reference_label": str(labels[i]),
        "amplitude_range": round(amplitude_range, 3),
        "largest_change": round(largest_change, 3),
        "quality_flags": reason,
        "system_status": decision,
    })

df = pd.DataFrame(results)

Path("results").mkdir(exist_ok=True)
df.to_csv("results/quality_aware_demo.csv", index=False)

print(df.to_string(index=False))
print("\nSaved to results/quality_aware_demo.csv")
print("\nREVIEWABLE does not mean AF was ruled out.")
print("This demonstration is not a medical diagnostic system.")