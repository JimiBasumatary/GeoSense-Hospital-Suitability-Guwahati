import pandas as pd
from pathlib import Path

print("=" * 60)
print("GeoSense - Creating ML Training Labels")
print("=" * 60)

# ---------------------------------------------------------
# 1. Define project paths
# ---------------------------------------------------------

project_root = Path(__file__).resolve().parents[2]

input_file = (
    project_root
    / "data"
    / "processed"
    / "hospital_suitability_dataset.csv"
)

output_file = (
    project_root
    / "data"
    / "processed"
    / "labelled_sites.csv"
)

# ---------------------------------------------------------
# 2. Load the Exercise 4 suitability dataset
# ---------------------------------------------------------

df = pd.read_csv(input_file)

print("\nInput dataset:")
print(input_file)

print("\nNumber of candidate locations:", len(df))

# ---------------------------------------------------------
# 3. Check the suitability classes
# ---------------------------------------------------------

print("\nExisting suitability classes:")
print(df["suitability_class"].value_counts())

# ---------------------------------------------------------
# 4. Convert suitability classes to ML labels
# ---------------------------------------------------------

label_mapping = {
    "Low": 0,
    "Moderate": 1,
    "High": 2
}

df["label"] = df["suitability_class"].map(label_mapping)

# ---------------------------------------------------------
# 5. Check for invalid/unmapped classes
# ---------------------------------------------------------

if df["label"].isna().any():
    print("\nERROR: Some suitability classes could not be mapped.")
    print(df.loc[df["label"].isna(), "suitability_class"].unique())
    raise ValueError("Invalid suitability class detected.")

df["label"] = df["label"].astype(int)

# ---------------------------------------------------------
# 6. Print label counts
# ---------------------------------------------------------

label_counts = df["label"].value_counts().sort_index()

print("\nTraining label counts:")
print(label_counts)

# ---------------------------------------------------------
# 7. Print label percentages
# ---------------------------------------------------------

label_percent = (
    df["label"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

print("\nTraining label percentages:")
print(label_percent.round(2))

# ---------------------------------------------------------
# 8. Save labelled dataset
# ---------------------------------------------------------

df.to_csv(output_file, index=False)

print("\nLabel mapping:")
print("0 = Low")
print("1 = Moderate")
print("2 = High")

print("\nLabelled dataset saved to:")
print(output_file)

print("\nTotal records:", len(df))

print("\n" + "=" * 60)
print("Training label creation completed successfully.")
print("=" * 60)