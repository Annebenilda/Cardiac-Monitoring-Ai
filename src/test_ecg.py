
from pathlib import Path

import wfdb
import matplotlib
matplotlib.use("Agg")  # Save the graph without opening a GUI window
import matplotlib.pyplot as plt

# Ensure the output folder exists
Path("results").mkdir(exist_ok=True)

# Read the first 10 seconds of ECG
record = wfdb.rdrecord(
    "04015",
    pn_dir="afdb",
    sampfrom=0,
    sampto=2500
)

ecg = record.p_signal[:, 0]
fs = record.fs

print("Sampling rate:", fs, "Hz")
print("Samples loaded:", len(ecg))
print("Duration:", len(ecg) / fs, "seconds")

# Plot and save the ECG
time = [i / fs for i in range(len(ecg))]

plt.figure(figsize=(12, 4))
plt.plot(time, ecg)
plt.xlabel("Time (seconds)")
plt.ylabel("ECG amplitude (mV)")
plt.title("First 10 seconds of ECG — record 04015")
plt.grid(True)
plt.tight_layout()

output_file = Path("results") / "first_ecg_plot.png"
plt.savefig(output_file, dpi=150)
plt.close()

print("ECG graph saved to:", output_file.resolve())