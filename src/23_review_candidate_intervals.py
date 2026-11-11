
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, find_peaks


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "results" / "expanded_ecg_dataset_clean.npz"
METADATA_PATH = PROJECT_DIR / "results" / "expanded_ecg_metadata_clean.csv"
CSV_PATH = PROJECT_DIR / "results" / "candidate_interval_review.csv"
PLOT_PATH = PROJECT_DIR / "results" / "candidate_interval_review.png"

FS = 250
MIN_DISTANCE_SEC = 0.30


def detect_candidates(signal):
    signal = np.asarray(signal, dtype=float)
    signal = signal - np.median(signal)

    sos = butter(3, [0.5, 40.0], btype="bandpass", fs=FS, output="sos")
    filtered = sosfiltfilt(sos, signal)

    distance = int(MIN_DISTANCE_SEC * FS)
    threshold = max(0.20 * np.std(filtered), 1e-6)

    options = []

    for polarity in (1, -1):
        peaks, properties = find_peaks(
            polarity * filtered,
            distance=distance,
            prominence=threshold
        )

        prominence = properties.get("prominences", np.array([]))
        rr = np.diff(peaks) / FS

        variability = (
            np.std(rr) / np.mean(rr)
            if len(rr) >= 2 and np.mean(rr) > 0
            else 10.0
        )

        score = (np.median(prominence) if len(prominence) else 0) - (
            0.15 * variability
        )

        options.append((score, peaks, polarity, filtered))

    _, peaks, polarity, filtered = max(options, key=lambda x: x[0])
    return filtered, peaks, polarity


def main():
    with np.load(DATA_PATH, allow_pickle=True) as data:
        signals = data["signals"]

    metadata = pd.read_csv(METADATA_PATH)

    if len(signals) != len(metadata):
        raise ValueError("Signal and metadata row counts do not match.")

    chosen_rows = []
    used_records = set()

    for label in ("AFIB", "NON_AFIB"):
        indices = metadata.index[
            metadata["label"].astype(str).str.upper() == label
        ].tolist()

        chosen = None
        for idx in indices:
            record = str(metadata.iloc[idx].get("record_id", "unknown"))
            if record not in used_records:
                chosen = idx
                used_records.add(record)
                break

        if chosen is None and indices:
            chosen = indices[0]

        if chosen is None:
            raise ValueError(f"No examples found for {label}")

        chosen_rows.append((label, chosen))

    records = []
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    for ax, (label, idx) in zip(axes, chosen_rows):
        filtered, peaks, polarity = detect_candidates(signals[idx])
        peak_times = peaks / FS
        intervals = np.diff(peak_times)

        row = metadata.iloc[idx]
        record_id = row.get("record_id", "unknown")

        for i, interval in enumerate(intervals):
            records.append({
                "label": label,
                "record_id": record_id,
                "interval_number": i + 1,
                "peak_time_1_sec": peak_times[i],
                "peak_time_2_sec": peak_times[i + 1],
                "interval_sec": interval,
                "interval_bpm_equivalent": 60.0 / interval,
                "flag_under_0_30_sec": interval < 0.30,
                "flag_over_2_sec": interval > 2.0,
            })

        ax.plot(peak_times[1:], intervals, marker="o", linewidth=1)
        ax.set_title(
            f"{label} | Record {record_id} | "
            f"{len(peaks)} candidates | "
            f"{'positive' if polarity == 1 else 'negative'} polarity"
        )
        ax.set_ylabel("Interval (seconds)")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time of second candidate (seconds)")
    fig.suptitle("Exploratory Candidate Intervals — Not Validated RR Intervals")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(records).to_csv(CSV_PATH, index=False)

    print(f"Saved interval table: {CSV_PATH}")
    print(f"Saved interval plot: {PLOT_PATH}")
    print("These are candidate-to-candidate intervals, not validated RR intervals.")


if __name__ == "__main__":
    main()
