
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# FILE 28: DUPLICATE AND OVERLAP AUDIT
# Reads the dataset; does not modify the original.
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"

METADATA_PATH = RESULTS_DIR / "expanded_ecg_metadata_clean.csv"

REPORT_PATH = RESULTS_DIR / "duplicate_overlap_audit.txt"
DUPLICATES_PATH = RESULTS_DIR / "duplicate_segment_starts.csv"
OVERLAPS_PATH = RESULTS_DIR / "overlapping_segments.csv"


def normalize_record_id(value):
    """Normalize IDs such as 04015 and 4015 to the same ID."""
    text = str(value).strip()

    # Remove a trailing .0 sometimes introduced by CSV numeric IDs.
    if text.endswith(".0"):
        text = text[:-2]

    # Numeric record IDs in this project can safely lose leading zeros.
    if text.isdigit():
        return str(int(text))

    return text


def main():
    print("=" * 60)
    print("ECG DUPLICATE AND OVERLAP AUDIT")
    print("=" * 60)

    if not METADATA_PATH.exists():
        print("\nERROR: Metadata file not found:")
        print(METADATA_PATH)
        return

    df = pd.read_csv(METADATA_PATH)

    print("\nOriginal metadata columns:")
    print(df.columns.tolist())

    # Support either column name.
    if "normalized_record_id" not in df.columns:
        if "record_id" in df.columns:
            df["normalized_record_id"] = df["record_id"].apply(
                normalize_record_id
            )
            print("\nCreated normalized_record_id from record_id.")
        else:
            print("\nERROR: No record_id column found.")
            return

    required = [
        "normalized_record_id",
        "start_seconds",
        "duration_seconds",
        "label",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        print("\nERROR: Missing required columns:", missing)
        print("Available columns:", df.columns.tolist())
        return

    df["start_seconds"] = pd.to_numeric(
        df["start_seconds"], errors="coerce"
    )
    df["duration_seconds"] = pd.to_numeric(
        df["duration_seconds"], errors="coerce"
    )

    report = [
        "ECG DUPLICATE AND OVERLAP AUDIT",
        "=" * 60,
        f"Metadata file: {METADATA_PATH}",
        f"Total segments: {len(df)}",
        f"Unique recordings: {df['normalized_record_id'].nunique()}",
        "",
    ]

    # 1. Duplicate start times in the same recording
    valid = df.dropna(
        subset=["normalized_record_id", "start_seconds"]
    ).copy()

    duplicate_mask = valid.duplicated(
        subset=["normalized_record_id", "start_seconds"],
        keep=False,
    )
    duplicates = valid.loc[duplicate_mask].sort_values(
        ["normalized_record_id", "start_seconds"]
    )

    duplicates.to_csv(DUPLICATES_PATH, index=False)

    report += ["1. DUPLICATE SEGMENT START TIMES", "-" * 60]
    if duplicates.empty:
        report.append("No repeated start times found.")
    else:
        report.append(f"Rows involved in repeated starts: {len(duplicates)}")
        report.append(duplicates.to_string(index=False))

    # 2. Overlapping intervals within each recording
    overlap_rows = []

    intervals = df.dropna(
        subset=[
            "normalized_record_id",
            "start_seconds",
            "duration_seconds",
        ]
    ).copy()

    intervals = intervals[intervals["duration_seconds"] > 0]

    for record_id, group in intervals.groupby("normalized_record_id"):
        group = group.sort_values("start_seconds")
        rows = list(group.iterrows())

        for i in range(len(rows)):
            idx_a, a = rows[i]
            start_a = float(a["start_seconds"])
            end_a = start_a + float(a["duration_seconds"])

            for j in range(i + 1, len(rows)):
                idx_b, b = rows[j]
                start_b = float(b["start_seconds"])

                if start_b >= end_a:
                    break

                end_b = start_b + float(b["duration_seconds"])
                overlap_seconds = min(end_a, end_b) - max(
                    start_a, start_b
                )

                if overlap_seconds > 0:
                    overlap_rows.append({
                        "normalized_record_id": record_id,
                        "row_a": idx_a,
                        "row_b": idx_b,
                        "start_a_seconds": start_a,
                        "end_a_seconds": end_a,
                        "label_a": a["label"],
                        "start_b_seconds": start_b,
                        "end_b_seconds": end_b,
                        "label_b": b["label"],
                        "overlap_seconds": overlap_seconds,
                    })

    overlap_columns = [
        "normalized_record_id", "row_a", "row_b",
        "start_a_seconds", "end_a_seconds", "label_a",
        "start_b_seconds", "end_b_seconds", "label_b",
        "overlap_seconds",
    ]
    overlaps = pd.DataFrame(overlap_rows, columns=overlap_columns)
    overlaps.to_csv(OVERLAPS_PATH, index=False)

    report += ["", "2. OVERLAPPING SEGMENTS", "-" * 60]
    if overlaps.empty:
        report.append("No overlapping time intervals found.")
    else:
        report.append(f"Overlapping segment pairs: {len(overlaps)}")
        report.append(overlaps.to_string(index=False))

    # 3. Timing checks
    report += ["", "3. TIMING DATA CHECK", "-" * 60]
    report.append(
        f"Missing/invalid start times: {df['start_seconds'].isna().sum()}"
    )
    report.append(
        f"Missing/invalid durations: {df['duration_seconds'].isna().sum()}"
    )
    report.append(
        "Non-positive durations: "
        f"{(df['duration_seconds'].dropna() <= 0).sum()}"
    )

    # 4. Recording totals
    report += ["", "4. SEGMENTS PER RECORDING", "-" * 60]
    counts = (
        df.groupby("normalized_record_id")
        .size()
        .rename("segment_count")
        .reset_index()
    )
    report.append(counts.to_string(index=False))

    report += [
        "",
        "INTERPRETATION",
        "-" * 60,
        "Repeated starts and overlaps are findings to review, not "
        "automatically proof of a data error.",
        "Adjacent segments that meet exactly at their boundaries "
        "are not counted as overlapping.",
        "This audit checks metadata timing, not waveform equality.",
    ]

    report_text = "\n".join(report)
    REPORT_PATH.write_text(report_text, encoding="utf-8")

    print("\n" + report_text)
    print("\nSaved report:", REPORT_PATH)
    print("Saved duplicate details:", DUPLICATES_PATH)
    print("Saved overlap details:", OVERLAPS_PATH)
    print("\nAudit complete. Original dataset unchanged.")


if __name__ == "__main__":
    main()
