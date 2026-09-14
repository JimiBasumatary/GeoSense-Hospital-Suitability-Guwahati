import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

print("=" * 60)
print("GeoSense - Training Label Distribution")
print("=" * 60)

# ---------------------------------------------------------
# 1. Define project paths
# ---------------------------------------------------------

project_root = Path(__file__).resolve().parents[2]

input_file = (
    project_root
    / "data"
    / "processed"
    / "labelled_sites.csv"
)

output_dir = (
    project_root
    / "outputs"
    / "reports"
)

output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "training_label_distribution.png"

# ---------------------------------------------------------
# 2. Load labelled dataset
# ---------------------------------------------------------

df = pd.read_csv(input_file)

print("\nDataset loaded successfully.")
print("Number of candidate locations:", len(df))

# ---------------------------------------------------------
# 3. Count suitability classes
# ---------------------------------------------------------

class_order = ["Low", "Moderate", "High"]

counts = (
    df["suitability_class"]
    .value_counts()
    .reindex(class_order)
)

print("\nTraining label distribution:")
print(counts)

# ---------------------------------------------------------
# 4. Create bar chart
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

bars = plt.bar(
    counts.index,
    counts.values
)

plt.xlabel("Suitability Class")
plt.ylabel("Number of Candidate Locations")
plt.title(
    "Training Label Distribution - "
    "Guwahati Hospital Suitability"
)

# Add values above bars
for bar, value in zip(bars, counts.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 10,
        str(value),
        ha="center"
    )

plt.tight_layout()

# ---------------------------------------------------------
# 5. Save the figure
# ---------------------------------------------------------

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nFigure saved to:")
print(output_file)

print("\nTraining label distribution plot created successfully.")

print("=" * 60)
print("Completed successfully.")
print("=" * 60)