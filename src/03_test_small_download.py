
from pathlib import Path

import wfdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

Path("results").mkdir(exist_ok=True)

print("Downloading only 10 seconds of ECG...")

record = wfdb.rdrecord(
    "04015",
    pn_dir="afdb",
    sampfrom=100 * 250,       # Start at 100 seconds
    sampto=110 * 250,         # Stop at 110 seconds
    channels=[0]
)

ecg = record.p_signal[:, 0]
fs = record.fs
time = [i / fs for i in range(len(ecg))]

plt.figure(figsize=(12, 4))
plt.plot(time, ecg)
plt.xlabel("Time within segment (seconds)")
plt.ylabel("ECG amplitude (mV)")
plt.title("ECG segment: 100–110 seconds")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/small_ecg_test.png", dpi=150)
plt.close()

print("Samples loaded:", len(ecg))
print("Graph saved to results/small_ecg_test.png")