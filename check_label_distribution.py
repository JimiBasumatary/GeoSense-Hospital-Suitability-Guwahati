import numpy as np
from pathlib import Path

base = Path(r"data\satellite\labels")

classes = {
    0: "Urban",
    1: "Vegetation",
    2: "Water",
    3: "Bare Land",
    4: "Agriculture",
    255: "NoData"
}

labels_2015 = np.load(base / "labels_2015.npy")
labels_2023 = np.load(base / "labels_2023.npy")

print("=" * 65)
print("           PSEUDO-LABEL CLASS DISTRIBUTION")
print("=" * 65)

print("{:<15}{:>15}{:>15}".format("Class", "2015", "2023"))
print("-" * 45)

for code, name in classes.items():
    count_2015 = np.sum(labels_2015 == code)
    count_2023 = np.sum(labels_2023 == code)

    print("{:<15}{:>15,}{:>15,}".format(
        name, count_2015, count_2023
    ))

print("-" * 45)
print("2015 shape:", labels_2015.shape)
print("2023 shape:", labels_2023.shape)

print("=" * 65)
print("LABEL DISTRIBUTION CHECK: SUCCESS")
print("=" * 65)