
import time
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

# ---------------- Configuration ----------------
FS = 250
DURATION_SECONDS = 10
SAMPLES = FS * DURATION_SECONDS
TARGET_SEGMENTS = 100
MAX_PER_RECORD = 10

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

DATASET_PATH = OUTPUT_DIR / "expanded_ecg_dataset.npz"
METADATA_PATH = OUTPUT_DIR / "expanded_ecg_metadata.csv"

# AFDB recordings with ECG signals.
RECORDS = [
    "04015", "04043", "04048", "04126", "04746",
    "04045", "04127", "04908", "04936", "05091",
    "05121", "05261", "06426", "06453", "06995",
    "07162", "07859", "07879", "07910", "08215",
    "08219", "08378", "08405",
]


def load_existing():
    """Resume from this script's previous output, if available."""
    if DATASET_PATH.exists() and METADATA_PATH.exists():
        data = np.load(DATASET_PATH, allow_pickle=True)
        signals = list(data["signals"])
        metadata = pd.read_csv(METADATA_PATH).to_dict("records")
        print(f"Resuming with {len(signals)} existing segments.")
        return signals, metadata

    return [], []


def save_dataset(signals, metadata):
    """Save after each successful segment so progress is retained."""
    if not signals:
        return

    df = pd.DataFrame(metadata)

    np.savez_compressed(
        DATASET_PATH,
        signals=np.asarray(signals, dtype=np.float32),
        labels=df["label"].to_numpy(dtype=str),
        record_ids=df["record_id"].to_numpy(dtype=str),
    )
    df.to_csv(METADATA_PATH, index=False)


def get_candidates(record_id):
    """Find annotated 10-second intervals fully within one rhythm label."""
    annotation = wfdb.rdann(record_id, "atr", pn_dir="afdb")
    changes = sorted(zip(annotation.sample, annotation.aux_note))

    candidates = []

    for i, (start, note) in enumerate(changes):
        note = note.strip()

        if note not in ("(AFIB", "(N"):
            continue

        end = (
            changes[i + 1][0]
            if i + 1 < len(changes)
            else float("inf")
        )

        # Start one second after the rhythm annotation boundary.
        segment_start = int(start) + FS
        segment_end = segment_start + SAMPLES

        if segment_end <= end:
            candidates.append({
                "start_sample": segment_start,
                "label": "AFIB" if note == "(AFIB" else "NON_AFIB",
                "reference_annotation": note,
            })

    return candidates


def main():
    signals, metadata = load_existing()

    existing_keys = {
        (str(row["record_id"]), int(row["start_seconds"]))
        for row in metadata
    }

    per_record = {}
    for row in metadata:
        rid = str(row["record_id"])
        per_record[rid] = per_record.get(rid, 0) + 1

    for record_id in RECORDS:
        if len(signals) >= TARGET_SEGMENTS:
            break

        if per_record.get(record_id, 0) >= MAX_PER_RECORD:
            continue

        print(f"\nChecking record {record_id}...")

        try:
            candidates = get_candidates(record_id)
        except Exception as exc:
            print(f"  Annotation read failed: {exc}")
            continue

        # Prefer the less represented class within this recording.
        class_counts = {"AFIB": 0, "NON_AFIB": 0}
        for row in metadata:
            if str(row["record_id"]) == record_id:
                class_counts[str(row["label"])] += 1

        candidates.sort(
            key=lambda item: (
                class_counts[item["label"]],
                item["start_sample"],
            )
        )

        added_here = 0

        for candidate in candidates:
            if len(signals) >= TARGET_SEGMENTS:
                break

            if per_record.get(record_id, 0) >= MAX_PER_RECORD:
                break

            start_sample = candidate["start_sample"]
            start_seconds = int(start_sample / FS)
            key = (record_id, start_seconds)

            if key in existing_keys:
                continue

            try:
                record = wfdb.rdrecord(
                    record_id,
                    pn_dir="afdb",
                    sampfrom=start_sample,
                    sampto=start_sample + SAMPLES,
                    channels=[0],
                )

                signal = record.p_signal[:, 0]

                if len(signal) != SAMPLES:
                    print("  Skipping incomplete segment.")
                    continue

                if not np.isfinite(signal).all():
                    print("  Skipping non-finite signal.")
                    continue

                signals.append(signal.astype(np.float32))
                metadata.append({
                    "record_id": record_id,
                    "start_seconds": start_seconds,
                    "duration_seconds": DURATION_SECONDS,
                    "label": candidate["label"],
                    "reference_annotation": (
                        candidate["reference_annotation"]
                    ),
                })

                existing_keys.add(key)
                per_record[record_id] = (
                    per_record.get(record_id, 0) + 1
                )
                class_counts[candidate["label"]] += 1
                added_here += 1

                # Persist each successful download.
                save_dataset(signals, metadata)

                print(
                    f"  Added {candidate['label']} at "
                    f"{start_seconds}s "
                    f"({len(signals)}/{TARGET_SEGMENTS})"
                )

                time.sleep(0.2)

            except Exception as exc:
                print(f"  Segment download failed: {exc}")

        if added_here == 0:
            print("  No new suitable segments added.")

    if not signals:
        raise RuntimeError(
            "No segments collected. Check your internet connection "
            "and AFDB record IDs."
        )

    df = pd.DataFrame(metadata)
    save_dataset(signals, metadata)

    print("\n=== Expanded dataset summary ===")
    print("Dataset shape:", np.asarray(signals).shape)
    print("Unique recordings:", df["record_id"].nunique())
    print("\nLabel counts:")
    print(df["label"].value_counts())
    print("\nSegments per recording:")
    print(df.groupby("record_id").size())
    print(f"\nSignals saved to: {DATASET_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")

    if len(signals) < TARGET_SEGMENTS:
        print(
            f"\nCollected {len(signals)} segments, below the target "
            f"of {TARGET_SEGMENTS}. The available annotated intervals "
            "or download failures may limit collection."
        )


if __name__ == "__main__":
    main()