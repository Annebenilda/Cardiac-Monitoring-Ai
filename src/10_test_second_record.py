
import wfdb

RECORD = "04043"
FS = 250
DURATION = 10
SAMPLES = FS * DURATION

print("Loading annotations for record", RECORD)

ann = wfdb.rdann(RECORD, "atr", pn_dir="afdb")

print("Annotation count:", len(ann.sample))
print("First 10 rhythm annotations:")

for sample, note in list(zip(ann.sample, ann.aux_note))[:10]:
    print(f"{sample / FS:.2f} seconds: {note!r}")

# Find a 10-second interval labelled normal or AF.
changes = sorted(zip(ann.sample, ann.aux_note))
chosen_start = None
chosen_label = None

for i, (start, note) in enumerate(changes):
    end = changes[i + 1][0] if i + 1 < len(changes) else float("inf")

    if note.strip() not in ("(N", "(AFIB"):
        continue

    # Pick a segment safely inside this annotated interval.
    candidate = max(start, 0)
    if candidate + SAMPLES <= end:
        chosen_start = candidate
        chosen_label = note.strip()
        break

if chosen_start is None:
    raise RuntimeError("No suitable 10-second interval found.")

print(f"\nReading signal at {chosen_start / FS:.2f} seconds")
record = wfdb.rdrecord(
    RECORD,
    pn_dir="afdb",
    sampfrom=int(chosen_start),
    sampto=int(chosen_start + SAMPLES),
    channels=[0]
)

print("Samples loaded:", len(record.p_signal))
print("Sampling rate:", record.fs, "Hz")
print("Reference rhythm:", chosen_label)