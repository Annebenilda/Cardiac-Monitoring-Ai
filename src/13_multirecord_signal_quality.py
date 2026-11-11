
import numpy as np
import pandas as pd
from pathlib import Path

# Configuration
SAMPLE_RATE = 250
FLATLINE_THRESHOLD_SECONDS = 1.0
NEAR_IDENTICAL_THRESHOLD = 1e-8

DATA_PATH = Path("results/multirecord_ecg_dataset.npz")
OUTPUT_PATH = Path("results/multirecord_signal_quality.csv")


def longest_consecutive_run(mask):
    """Return the longest consecutive run of True values."""
    longest_run = 0
    current_run = 0

    for value in mask:
        if value:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0

    return longest_run


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run src/11_build_multirecord_dataset.py first."
        )

    # Load the locally generated dataset.
    data = np.load(DATA_PATH, allow_pickle=True)

    signals = data["signals"]
    labels = data["labels"]
    record_ids = data["record_ids"]

    if signals.ndim != 2:
        raise ValueError(
            f"Expected a 2D signal array; got shape {signals.shape}"
        )

    if not (len(signals) == len(labels) == len(record_ids)):
        raise ValueError(
            "Signals, labels, and record IDs have different lengths."
        )

    results = []

    for i, signal in enumerate(signals):
        signal = np.asarray(signal, dtype=float)

        record_id = str(record_ids[i])
        label = str(labels[i])

        flags = []

        # Handle invalid values safely.
        if not np.isfinite(signal).all():
            results.append({
                "record_id": record_id,
                "label": label,
                "amplitude_range": np.nan,
                "signal_std": np.nan,
                "largest_step": np.nan,
                "flat_fraction": np.nan,
                "longest_flat_seconds": np.nan,
                "quality_status": "REVIEW",
                "quality_flags": "non_finite_values",
            })
            continue

        # Basic signal measurements.
        amplitude_range = float(np.ptp(signal))
        signal_std = float(np.std(signal))

        differences = np.abs(np.diff(signal))

        largest_step = (
            float(np.max(differences))
            if len(differences) > 0
            else 0.0
        )

        # Fraction of adjacent sample pairs with very small changes.
        flat_fraction = (
            float(np.mean(differences < NEAR_IDENTICAL_THRESHOLD))
            if len(differences) > 0
            else 0.0
        )

        # Find sustained runs of nearly identical samples.
        near_flat = differences < NEAR_IDENTICAL_THRESHOLD
        longest_run = longest_consecutive_run(near_flat)
        longest_flat_seconds = longest_run / SAMPLE_RATE

        # Initial engineering heuristics only; not clinical thresholds.
        if amplitude_range > 6.0:
            flags.append("large_amplitude_range")

        if largest_step > 2.0:
            flags.append("abrupt_change")

        if longest_flat_seconds >= FLATLINE_THRESHOLD_SECONDS:
            flags.append("possible_flatline")

        quality_status = (
            "REVIEW" if flags else "NO_RULE_TRIGGERED"
        )

        results.append({
            "record_id": record_id,
            "label": label,
            "amplitude_range": amplitude_range,
            "signal_std": signal_std,
            "largest_step": largest_step,
            "flat_fraction": flat_fraction,
            "longest_flat_seconds": longest_flat_seconds,
            "quality_status": quality_status,
            "quality_flags": "; ".join(flags) if flags else "none",
        })

    # Save report.
    df = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("=== Multi-record ECG signal quality report ===")
    print(df.to_string(index=False))

    print("\nQuality status counts:")
    print(df["quality_status"].value_counts())

    print("\nSegments flagged for review:")
    flagged = df[df["quality_status"] == "REVIEW"]

    if flagged.empty:
        print("No segments triggered the current rules.")
    else:
        print(
            flagged[
                ["record_id", "label", "quality_flags"]
            ].to_string(index=False)
        )

    print(f"\nReport saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()