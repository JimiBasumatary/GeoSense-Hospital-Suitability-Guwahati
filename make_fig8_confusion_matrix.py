
import os
import numpy as np
import matplotlib.pyplot as plt

# Confusion-matrix counts transcribed from your existing Guwahati figure.
# Rows = actual classes; columns = predicted classes.
cm = np.array([
    [0, 68, 0, 1215, 2998],
    [0, 358416, 0, 647, 21461],
    [102, 627, 14, 26435, 5693],
    [65, 3053, 1, 25852, 63253],
    [66, 67581, 0, 785, 160839]
], dtype=np.int64)

classes = [
    "Urban",
    "Vegetation",
    "Water",
    "Bare Land",
    "Agriculture"
]

# Convert each row to percentages.
row_totals = cm.sum(axis=1, keepdims=True)
cm_percent = np.divide(
    cm.astype(float) * 100,
    row_totals,
    out=np.zeros_like(cm, dtype=float),
    where=row_totals != 0
)

# Plot the row-normalised confusion matrix.
fig, ax = plt.subplots(figsize=(10, 8))

im = ax.imshow(
    cm_percent,
    cmap="Blues",
    vmin=0,
    vmax=100,
    aspect="auto"
)

ax.set_title(
    "U-Net Row-Normalised Confusion Matrix - Guwahati\n"
    "(Percentages; source dataset must be verified)",
    fontsize=14,
    pad=16
)
ax.set_xlabel("Predicted Class", fontsize=12)
ax.set_ylabel("Actual Class", fontsize=12)

ax.set_xticks(range(len(classes)))
ax.set_xticklabels(classes, rotation=30, ha="right")
ax.set_yticks(range(len(classes)))
ax.set_yticklabels(classes)

# Annotate each cell with its row percentage.
for i in range(len(classes)):
    for j in range(len(classes)):
        value = cm_percent[i, j]
        text_color = "white" if value >= 50 else "black"
        ax.text(
            j, i, f"{value:.1f}%",
            ha="center", va="center",
            color=text_color, fontsize=10
        )

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Percentage of actual-class pixels (%)")

fig.tight_layout()

output_dir = r"D:\GeoSense_Agent\outputs\plots"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(
    output_dir,
    "unet_confusion_matrix_row_normalized.png"
)

fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Row-normalised confusion matrix created:")
print(output_path)
print("\nRow totals:")
for name, total in zip(classes, row_totals.ravel()):
    print(f"{name}: {total:,} pixels")

print("\nNote: This figure uses counts copied from the existing image.")
print("Verify whether those counts represent validation or test data.")
