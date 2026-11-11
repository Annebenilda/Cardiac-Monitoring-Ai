
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = (
    PROJECT_DIR / "results" / "grouped_baseline_predictions.csv"
)
METRICS_PATH = PROJECT_DIR / "results" / "grouped_baseline_metrics.txt"
OUTPUT_PATH = PROJECT_DIR / "results" / "baseline_audit_report.txt"


def find_column(df, possibilities):
    """Find a column without depending on capitalization."""
    lookup = {str(col).strip().lower(): col for col in df.columns}
    for name in possibilities:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def main():
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_PATH}"
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    record_col = find_column(df, ["record_id", "record", "group"])
    true_col = find_column(
        df, ["true_label", "label", "actual_label", "y_true"]
    )
    pred_col = find_column(
        df, ["predicted_label", "prediction", "pred_label", "y_pred"]
    )

    if true_col is None or pred_col is None:
        raise ValueError(
            "Could not identify true and predicted label columns.\n"
            f"Found columns: {list(df.columns)}"
        )

    if record_col is None:
        raise ValueError(
            "Could not identify the recording ID column.\n"
            f"Found columns: {list(df.columns)}"
        )

    df = df.dropna(subset=[true_col, pred_col, record_col]).copy()
    df[true_col] = df[true_col].astype(str).str.strip().str.upper()
    df[pred_col] = df[pred_col].astype(str).str.strip().str.upper()
    df[record_col] = df[record_col].astype(str).str.strip()

    labels = ["AFIB", "NON_AFIB"]
    unknown_true = sorted(set(df[true_col]) - set(labels))
    unknown_pred = sorted(set(df[pred_col]) - set(labels))

    if unknown_true or unknown_pred:
        raise ValueError(
            f"Unexpected labels. True: {unknown_true}; "
            f"predicted: {unknown_pred}"
        )

    y_true = df[true_col]
    y_pred = df[pred_col]

    report_lines = []
    report_lines.append("GROUPED BASELINE AUDIT")
    report_lines.append("=" * 60)
    report_lines.append(f"Prediction rows: {len(df)}")
    report_lines.append(f"Unique recordings: {df[record_col].nunique()}")
    report_lines.append("True class counts:")
    report_lines.append(y_true.value_counts().to_string())
    report_lines.append("")
    report_lines.append(
        f"Accuracy: {accuracy_score(y_true, y_pred):.3f}"
    )
    report_lines.append(
        "Balanced accuracy: "
        f"{balanced_accuracy_score(y_true, y_pred):.3f}"
    )
    report_lines.append("")
    report_lines.append("Classification report:")
    report_lines.append(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
            digits=3,
        )
    )
    report_lines.append("Confusion matrix (rows=true, columns=predicted):")
    report_lines.append(
        pd.DataFrame(
            confusion_matrix(y_true, y_pred, labels=labels),
            index=[f"True {x}" for x in labels],
            columns=[f"Pred {x}" for x in labels],
        ).to_string()
    )

    report_lines.append("")
    report_lines.append("PER-RECORD PERFORMANCE")
    report_lines.append("-" * 60)

    per_record = (
        df.assign(correct=(y_true == y_pred))
        .groupby(record_col)
        .agg(
            segments=(true_col, "size"),
            correct=("correct", "sum"),
            accuracy=("correct", "mean"),
            true_afib=(true_col, lambda s: (s == "AFIB").sum()),
            true_non_afib=(true_col, lambda s: (s == "NON_AFIB").sum()),
        )
        .sort_index()
    )
    report_lines.append(per_record.to_string(float_format=lambda x: f"{x:.3f}"))

    report_lines.append("")
    report_lines.append("DUPLICATE CHECK")
    report_lines.append("-" * 60)

    # If available, check whether a recording has multiple identical
    # prediction rows. This is a warning check, not proof of leakage.
    start_col = find_column(
        df, ["start_sec", "start_time_sec", "start_time", "segment_start"]
    )
    duplicate_cols = [record_col]
    if start_col is not None:
        duplicate_cols.append(start_col)
    duplicate_cols.extend([true_col, pred_col])

    duplicates = df.duplicated(subset=duplicate_cols, keep=False)
    report_lines.append(
        f"Rows in repeated record/time/label/prediction groups: "
        f"{int(duplicates.sum())}"
    )
    if start_col is None:
        report_lines.append(
            "No recognized segment-start column was found; "
            "duplicate check is limited."
        )

    report_lines.append("")
    report_lines.append("INTERPRETATION")
    report_lines.append("-" * 60)
    report_lines.append(
        "This is an exploratory audit of saved predictions, not an "
        "independent clinical validation."
    )
    report_lines.append(
        "Group separation by recording reduces segment-level leakage, "
        "but does not establish generalization to new patients or devices."
    )
    report_lines.append(
        "Repeated segments or adjacent segments from one recording may "
        "still affect independence; inspect dataset construction."
    )

    report = "\n".join(report_lines)
    print(report)

    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(f"\nSaved audit report: {OUTPUT_PATH}")

    if METRICS_PATH.exists():
        print(f"\nPreviously saved metrics file: {METRICS_PATH}")
        print("The existing metrics file was not modified.")


if __name__ == "__main__":
    main()
