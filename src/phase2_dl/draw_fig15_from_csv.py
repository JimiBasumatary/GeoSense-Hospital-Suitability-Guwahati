
import os
import pandas as pd
import matplotlib.pyplot as plt

root = r"D:\GeoSense_Agent"
csv_file = os.path.join(
    root, "outputs", "plots", "fig15_per_chip_accuracy.csv"
)
output_file = os.path.join(
    root, "outputs", "plots",
    "fig15_per_chip_accuracy_comparison.png"
)

df = pd.read_csv(csv_file)

required = [
    "unet_accuracy_percent",
    "substitute_accuracy_percent",
]
if not all(col in df.columns for col in required):
    raise ValueError("Expected accuracy columns were not found in CSV.")

x = df["unet_accuracy_percent"]
y = df["substitute_accuracy_percent"]

fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(x, y, s=65, alpha=0.85)
ax.plot([0, 100], [0, 100], "--", linewidth=1.5)

ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("U-Net pixel accuracy (%)")
ax.set_ylabel("ResNet-50 U-Net substitute pixel accuracy (%)")
ax.set_title("Per-Chip Pixel Accuracy — Guwahati Held-out Chips")
ax.grid(True, alpha=0.25)

fig.text(
    0.5, 0.015,
    f"Chips plotted: {len(df)} | NoData label 255 excluded",
    ha="center", fontsize=9
)
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(output_file, dpi=300, bbox_inches="tight")
plt.close(fig)

print("FIG. 15 PLOT SAVED:", output_file)
print("Chips plotted:", len(df))
