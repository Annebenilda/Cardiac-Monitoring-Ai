
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

# --------------------------------------------------
# CARDIAC MONITORING RESEARCH DEMO
# Not a medical device or diagnostic tool.
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
RESULTS_DIR = PROJECT_DIR / "results"

DATA_PATH = RESULTS_DIR / "expanded_ecg_dataset_clean.npz"
METADATA_PATH = RESULTS_DIR / "expanded_ecg_metadata_clean.csv"
MODEL_PATH = RESULTS_DIR / "cardiac_demo_model.joblib"
FEATURE_SCRIPT = SRC_DIR / "30_train_demo_model.py"

FS = 250

st.set_page_config(
    page_title="Cardiac Monitoring AI",
    page_icon="💓",
    layout="wide",
)

st.title("💓 AI-Powered Cardiac Monitoring")
st.subheader("Experimental Arrhythmia Screening Prototype")

st.warning(
    "RESEARCH DEMO ONLY. This prototype is not a medical device, "
    "does not provide a diagnosis, and must not be used to make "
    "healthcare decisions."
)

# Load dataset
if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

try:
    with np.load(DATA_PATH, allow_pickle=True) as data:
        signals = data["signals"]
        labels = data["labels"].astype(str)
        record_ids = data["record_ids"].astype(str)
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

if not (len(signals) == len(labels) == len(record_ids)):
    st.error("Dataset arrays have different lengths.")
    st.stop()

if signals.ndim != 2 or signals.shape[1] == 0:
    st.error("Unexpected signal array shape.")
    st.stop()

metadata = None
if METADATA_PATH.exists():
    try:
        candidate = pd.read_csv(METADATA_PATH)
        if len(candidate) == len(signals):
            metadata = candidate
    except Exception:
        metadata = None

# Load feature extraction function used for training.
extract_features = None
try:
    spec = importlib.util.spec_from_file_location(
        "demo_training_module", FEATURE_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load feature-extraction script.")

    feature_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(feature_module)
    extract_features = feature_module.extract_features
except Exception as exc:
    st.error(f"Could not load feature extraction: {exc}")
    st.stop()

# Load saved model, if training has been completed.
model_bundle = None
model_error = None

if MODEL_PATH.exists():
    try:
        # Only load this project-created file that you trust.
        model_bundle = joblib.load(MODEL_PATH)

        if not isinstance(model_bundle, dict):
            raise ValueError("Unexpected model file format.")
        if "model" not in model_bundle or "feature_columns" not in model_bundle:
            raise ValueError("Model file is missing required fields.")
        if model_bundle.get("sampling_frequency") != FS:
            raise ValueError("Model sampling frequency does not match.")
    except Exception as exc:
        model_error = str(exc)
        model_bundle = None

# Dataset summary
c1, c2, c3, c4 = st.columns(4)
c1.metric("ECG segments", len(signals))
c2.metric("Recordings", len(np.unique(record_ids)))
c3.metric("AFIB-labeled segments", int(np.sum(labels == "AFIB")))
c4.metric("Other-labeled segments", int(np.sum(labels == "NON_AFIB")))

st.divider()
st.header("1. Explore an ECG segment")

recording_options = sorted(np.unique(record_ids).tolist())
selected_record = st.selectbox("Choose a recording", recording_options)

available_indices = np.where(record_ids == selected_record)[0]
selected_index = st.selectbox(
    "Choose a segment",
    available_indices.tolist(),
    format_func=lambda i: f"Segment {i + 1} — reference label: {labels[i]}",
)

signal = np.asarray(signals[selected_index], dtype=float)

if not np.isfinite(signal).all():
    st.error("This segment contains non-finite samples.")
    st.stop()

time = np.arange(len(signal)) / FS

left, right = st.columns([2, 1])

with left:
    st.markdown("**ECG waveform**")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(time, signal, linewidth=0.8)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Signal amplitude (dataset units)")
    ax.set_title(f"Recording {selected_record} — segment {selected_index + 1}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.markdown("**Segment information**")
    st.write(f"Array index: {selected_index}")
    st.write(f"Recording ID: {selected_record}")
    st.write(f"Samples: {len(signal)}")
    st.write(f"Duration: {len(signal) / FS:.1f} seconds")
    st.write(f"Sampling frequency: {FS} Hz")

    st.markdown("**Basic signal measurements**")
    st.write(f"Minimum: {np.min(signal):.4f}")
    st.write(f"Maximum: {np.max(signal):.4f}")
    st.write(f"Standard deviation: {np.std(signal):.4f}")
    st.write(f"RMS amplitude: {np.sqrt(np.mean(signal ** 2)):.4f}")

st.divider()
st.header("2. Basic signal-quality checks")

signal_range = float(np.ptp(signal))
signal_std = float(np.std(signal))
max_step = (
    float(np.max(np.abs(np.diff(signal))))
    if len(signal) > 1 else 0.0
)

flat_steps = np.abs(np.diff(signal)) < 1e-8
longest_flat_run = 0
current_run = 0

for is_flat in flat_steps:
    if is_flat:
        current_run += 1
        longest_flat_run = max(longest_flat_run, current_run)
    else:
        current_run = 0

flags = []

if signal_std < 1e-6:
    flags.append("Nearly constant signal")
if signal_range > 6:
    flags.append("Large amplitude range — review waveform")
if max_step > 2:
    flags.append("Large sample-to-sample change — review waveform")
if longest_flat_run >= FS:
    flags.append("Possible flatline-like interval — review waveform")

if flags:
    st.warning("Review recommended based on simple experimental rules.")
    for flag in flags:
        st.write(f"- {flag}")
else:
    st.info(
        "No configured quality rule was triggered. This does not "
        "guarantee that the ECG is good quality."
    )

st.caption(
    "These are heuristic checks, not validated clinical "
    "signal-quality measurements."
)

st.divider()
st.header("3. Experimental reference label")

reference_label = str(labels[selected_index])
st.write(f"**Dataset reference label:** {reference_label}")

st.info(
    "The reference label comes from the dataset, not from the AI. "
    "NON_AFIB means the segment was not labeled AFIB in this dataset; "
    "it does not guarantee normal sinus rhythm."
)

st.subheader("Experimental AI screening result")

if model_error:
    st.error(f"The saved model could not be loaded: {model_error}")
elif model_bundle is None:
    st.warning(
        "No saved demo model was found. Run "
        "`python src/30_train_demo_model.py` from your project folder "
        "to create it, then refresh this page."
    )
else:
    try:
        feature_values = extract_features(signal)
        feature_columns = model_bundle["feature_columns"]

        feature_row = pd.DataFrame([feature_values])
        feature_row = feature_row.reindex(columns=feature_columns)
        feature_row = feature_row.replace([np.inf, -np.inf], np.nan)

        if feature_row.isna().all(axis=1).iloc[0]:
            raise ValueError("All extracted features are missing.")

        model = model_bundle["model"]
        prediction = str(model.predict(feature_row)[0])

        st.write(f"**Experimental model prediction:** {prediction}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(feature_row)[0]
            classes = list(model.named_steps["logisticregression"].classes_)

            if "AFIB" in classes:
                afib_score = float(probabilities[classes.index("AFIB")])
                st.metric(
                    "Model AFIB score",
                    f"{afib_score:.1%}",
                )
                st.caption(
                    "This is an uncalibrated model score, not a validated "
                    "clinical probability or diagnostic confidence."
                )

        if prediction == reference_label:
            st.info("The experimental prediction matches the dataset reference label.")
        else:
            st.warning("The experimental prediction differs from the dataset reference label.")

        st.caption(
            "This model was fitted on the full cleaned dataset. The selected "
            "segment may have been used during training, so this comparison "
            "is not an independent test of model performance."
        )

    except Exception as exc:
        st.error(f"Could not generate an experimental prediction: {exc}")

st.divider()
st.header("4. Project evaluation")

st.write(
    "The separate grouped baseline evaluation reported 59.5% accuracy "
    "and 60.5% balanced accuracy across eight recordings."
)

st.write(
    "These preliminary results are limited and are not suitable for "
    "clinical use. The interactive demo model is trained on the full "
    "dataset and its selected-segment predictions must not be treated "
    "as held-out evaluation results."
)

st.caption(
    "Dataset: MIT-BIH Atrial Fibrillation Database. "
    "Research and educational prototype only."
)
