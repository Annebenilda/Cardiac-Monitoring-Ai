
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
METADATA_PATH = (
    PROJECT_DIR / "results" / "expanded_ecg_metadata_clean.csv"
)
OUTPUT_PATH = PROJECT_DIR / "results" / "segment_distribution_audit.txt"


def normalize_record_id(value):
    value = str(value).strip()
    if value.endswith(".0"):
        value = value[:-2]
    if value.isdigit():
        return value.lstrip("0") or "0"
    return value


def main():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata not found: {METADATA_PATH}")

    df = pd.read_csv(METADATA_PATH)

    required = {"record_id", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    df["normalized_record_id"] = df["record_id"].map(normalize_record_id)
    df["label"] = df["label"].astype(str).str.strip().str.upper()

    print("METADATA COLUMNS:")
    print(df.columns.tolist())

    print("\nOVERALL CLASS COUNTS:")
    print(df["label"].value_counts(dropna=False).to_string())

    print("\nSEGMENTS PER RECORDING AND LABEL:")
    distribution = pd.crosstab(
        df["normalized_record_id"],
        df["label"],
        margins=True,
    )
    print(distribution.to_string())

    print("\nRECORDING TOTALS:")
    totals = (
        df.groupby("normalized_record_id")
        .size()
        .rename("segment_count")
        .sort_index()
    )
    print(totals.to_string())

    print("\nPOSSIBLE DUPLICATE SEGMENT STARTS:")
    start_columns = [
        "start_sec",
        "start_time_sec",
        "start_time",
        "segment_start",
    ]
    start_col = next(
        (col for col in start_columns if col in df.columns),
        None,
    )

    if start_col is None:
        print("No recognized segment start-time column found.")
    else:
        duplicate_mask = df.duplicated(
            subset=["normalized_record_id", start_col],
            keep=False,
        )
        print(f"Rows sharing a recording and start time: {duplicate_mask.sum()}")
        if duplicate_mask.any():
            cols = ["record_id", "normalized_record_id", start_col, "label"]
            print(df.loc[duplicate_mask, cols].sort_values(
                ["normalized_record_id", start_col]
            ).to_string(index=False))

    report = [
        "SEGMENT DISTRIBUTION AUDIT",
        "=" * 60,
        "Metadata columns:",
        str(df.columns.tolist()),
        "",
        "Class counts:",
        df["label"].value_counts(dropna=False).to_string(),
        "",
        "Segments by recording and label:",
        distribution.to_string(),
        "",
        "Segments per recording:",
        totals.to_string(),
        "",
        "Duplicate-start check:",
        (
            f"Rows sharing a recording and start time: {int(duplicate_mask.sum())}"
            if start_col is not None
            else "No recognized start-time column was found."
        ),
        "",
        "This audit checks metadata consistency only. It does not verify "
        "the clinical correctness of rhythm annotations.",
    ]

    OUTPUT_PATH.write_text("\n".join(report), encoding="utf-8")
    print(f"\nSaved audit: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
