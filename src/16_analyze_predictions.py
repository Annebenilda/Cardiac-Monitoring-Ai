
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"

INPUT_PATH = RESULTS_DIR / "grouped_baseline_predictions.csv"
SUMMARY_PATH = RESULTS_DIR / "grouped_per_recording_analysis.csv"
ERRORS_PATH = RESULTS_DIR / "grouped_misclassified_segments.csv"
REPORT_PATH = RESULTS_DIR / "grouped_error_analysis_report.txt"

POSITIVE_LABEL = "AFIB"
NEGATIVE_LABEL = "NON_AFIB"


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_PATH}. "
            "Run the grouped baseline evaluation first."
        )

    df = pd.read_csv(INPUT_PATH)

    required = {
        "record_id",
        "true_label",
        "predicted_label",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df["record_id"] = df["record_id"].astype(str).str.strip()
    df["true_label"] = df["true_label"].astype(str).str.strip().str.upper()
    df["predicted_label"] = (
        df["predicted_label"].astype(str).str.strip().str.upper()
    )

    df["correct"] = df["true_label"] == df["predicted_label"]
    df["result"] = df["correct"].map({
        True: "CORRECT",
        False: "MISCLASSIFIED",
    })

    if "estimated_afib_probability" in df.columns:
        df["estimated_afib_probability"] = pd.to_numeric(
            df["estimated_afib_probability"],
            errors="coerce",
        )

    y_true = df["true_label"]
    y_pred = df["predicted_label"]

    print("\n=== Overall grouped evaluation ===")
    print(f"Segments: {len(df)}")
    print(f"Recordings: {df['record_id'].nunique()}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print(
        "Balanced accuracy: "
        f"{balanced_accuracy_score(y_true, y_pred):.3f}"
    )
    
    print(
        f"AFIB precision: "
        f"{precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0):.3f}"
    )
    print(
        f"AFIB recall: "
        f"{recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0):.3f}"
    )
    print(
        f"AFIB F1: "
        f"{f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0):.3f}"
    )

    rows = []

    for record_id, group in df.groupby("record_id", sort=True):
        truth = group["true_label"]
        pred = group["predicted_label"]

        rows.append({
            "record_id": record_id,
            "segments": len(group),
            "correct": int(group["correct"].sum()),
            "misclassified": int((~group["correct"]).sum()),
            "accuracy": accuracy_score(truth, pred),
            "balanced_accuracy": (
                balanced_accuracy_score(truth, pred)
                if truth.nunique() == 2 else float("nan")
            ),
            "afib_precision": precision_score(
                truth, pred, pos_label=POSITIVE_LABEL, zero_division=0
            ),
            "afib_recall": recall_score(
                truth, pred, pos_label=POSITIVE_LABEL, zero_division=0
            ),
            "afib_f1": f1_score(
                truth, pred, pos_label=POSITIVE_LABEL, zero_division=0
            ),
            "true_afib": int((truth == POSITIVE_LABEL).sum()),
            "true_non_afib": int((truth == NEGATIVE_LABEL).sum()),
            "predicted_afib": int((pred == POSITIVE_LABEL).sum()),
            "predicted_non_afib": int((pred == NEGATIVE_LABEL).sum()),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_PATH, index=False)

    errors = df.loc[~df["correct"]].copy()
    errors.to_csv(ERRORS_PATH, index=False)

    report = [
        "GROUPED PER-RECORDING ERROR ANALYSIS",
        "=" * 48,
        f"Segments evaluated: {len(df)}",
        f"Recordings: {df['record_id'].nunique()}",
        f"Correct: {int(df['correct'].sum())}",
        f"Misclassified: {len(errors)}",
        "",
        "Per-recording metrics:",
        summary.to_string(index=False),
        "",
        "Misclassified segments:",
        errors.to_string(index=False) if not errors.empty else "None",
        "",
        "Research prototype only; not for clinical use.",
    ]

    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    print("\n=== Per-recording results ===")
    print(summary.round(3).to_string(index=False))

    print("\n=== Misclassified segments ===")
    if errors.empty:
        print("No misclassified segments.")
    else:
        print(errors.to_string(index=False))

    print("\nSaved:")
    print(SUMMARY_PATH)
    print(ERRORS_PATH)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
