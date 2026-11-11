
import wfdb

record_name = "04015"

# Load the reference rhythm annotations
annotation = wfdb.rdann(
    record_name,
    "atr",
    pn_dir="afdb"
)

print("Record:", record_name)
print("Total annotations:", len(annotation.sample))
print("\nFirst 20 annotations:")

for i in range(min(20, len(annotation.sample))):
    sample = annotation.sample[i]
    time_seconds = sample / 250

    print(
        f"Annotation {i + 1}: "
        f"time={time_seconds:.2f} seconds, "
        f"symbol={annotation.symbol[i]!r}, "
        f"note={annotation.aux_note[i]!r}"
    )