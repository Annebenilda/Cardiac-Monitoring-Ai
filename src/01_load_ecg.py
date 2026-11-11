
from pathlib import Path

import matplotlib.pyplot as plt
import wfdb

# Create the local data folder.
data_dir = Path("data")
data_dir.mkdir(parents=True, exist_ok=True)

# Load one ECG record from PhysioNet.
record_name = "04015"

record = wfdb.rdrecord(
    record_name,
    pn_dir="afdb"
)

# Display basic information.
ecg = record.p_signal
sampling_rate = record.fs

print("Record:", record_name)
print("Sampling rate:", sampling_rate, "Hz")
print("Signal shape:", ecg.shape)
print("Channels:", record.sig_name)
print("Duration:", round(len(ecg) / sampling_rate, 2), "seconds")

# Plot the first 10 seconds of the first ECG channel.
seconds_to_plot = 10
sample_count = min(
    int(seconds_to_plot * sampling_rate),
    len(ecg)
)

time = (
    range(sample_count)
)

plt.figure(figsize=(12, 4))
plt.plot(
    [i / sampling_rate for i in time],
    ecg[:sample_count, 0]
)

plt.xlabel("Time (seconds)")
plt.ylabel("ECG amplitude (mV)")
plt.title(f"ECG waveform — record {record_name}")
plt.grid(True)
plt.tight_layout()

# Save the plot for your project report.
plt.savefig(
    "results/first_ecg_plot.png",
    dpi=150
)

plt.show()