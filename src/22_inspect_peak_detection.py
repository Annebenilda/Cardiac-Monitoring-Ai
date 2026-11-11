
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
OUTPUT_PATH = PROJECT_DIR / "results" / "improved_peak_detection_examples_v2.png"

FS = 250
MIN_DISTANCE_SEC = 0.30


def filter_ecg(signal):
    signal = np.asarray(signal, dtype=float)
    signal = signal - np.median(signal)

    sos = butter(
        3, [0.5, 40.0], btype="bandpass", fs=FS, output="sos"
    )
    return sosfiltfilt(sos, signal)


def get_candidates(filtered, polarity):
    oriented = polarity * filtered
    min_distance = int(MIN_DISTANCE_SEC * FS)

    # Adaptive prominence threshold; heuristic only.
    prominence = max(0.20 * np.std(filtered), 1e-6)

    peaks, properties = find_peaks(
        oriented,
        distance=min_distance,
        prominence=prominence
    )

    prominences = properties.get("prominences", np.array([]))

    # Reward strong peaks, but penalize implausibly irregular spacing.
    if len(peaks) >= 3:
        rr = np.diff(peaks) / FS
        rr_cv = np.std(rr) / max(np.mean(rr), 1e-6)
    else:
        rr_cv = 10.0

    score = (
        np.median(prominences) if len(prominences) else 0.0
    ) - 0.15 * rr_cv

    return peaks, score, rr_cv


def main():
    if not DATA_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Clean dataset or metadata is missing. "
            "Check results/expanded_ecg_dataset_clean.npz and "
            "results/expanded_ecg_metadata_clean.csv."
        )

    with np.load(DATA_PATH, allow_pickle=True) as data:
        signals = data["signals"]

    metadata = pd.read_csv(METADATA_PATH)

    if len(signals) != len(metadata):
        raise ValueError("Signal and metadata row counts do not match.")

    selected = []
    used_records = set()

    for label in ("AFIB", "NON_AFIB"):
        matching = metadata.index[
            metadata["label"].astype(str).str.upper() == label
        ].tolist()

        chosen = None
        for idx in matching:
            record = str(metadata.iloc[idx].get("record_id", "unknown"))
            if record not in used_records:
                chosen = idx
                used_records.add(record)
                break

        if chosen is None and matching:
            chosen = matching[0]

        if chosen is None:
            raise ValueError(f"No examples found for {label}")

        selected.append((label, chosen))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for ax, (label, idx) in zip(axes, selected):
        row = metadata.iloc[idx]
        filtered = filter_ecg(signals[idx])
        time = np.arange(len(filtered)) / FS

        positive, positive_score, positive_cv = get_candidates(filtered, 1)
        negative, negative_score, negative_cv = get_candidates(filtered, -1)

        # Choose the candidate set with the higher heuristic score.
        # This is not a clinically validated QRS selection method.
        if positive_score >= negative_score:
            peaks = positive
            polarity = "positive"
            chosen_cv = positive_cv
        else:
            peaks = negative
            polarity = "negative"
            chosen_cv = negative_cv

        ax.plot(time, filtered, linewidth=0.8, label="Filtered ECG")
        ax.scatter(
            time[peaks],
            filtered[peaks],
            marker="x",
            s=45,
            label="Selected candidates"
        )

        record = row.get("record_id", "unknown")
        start = row.get("start_sec", row.get("start_time_sec", "unknown"))

        ax.set_title(
            f"{label} | Record {record} | Start {start} s | "
            f"Candidates: {len(peaks)} | Polarity: {polarity} | "
            f"RR variability score: {chosen_cv:.2f}"
        )
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle(
        "Experimental ECG Candidate Detection — Manual Review Required"
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {OUTPUT_PATH}")
    print("This heuristic may still select incorrect peaks.")
    print("Do not treat candidate intervals as validated heartbeat intervals.")


if __name__ == "__main__":
    main()
