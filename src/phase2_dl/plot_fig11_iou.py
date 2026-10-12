
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(r"D:\GeoSense_Agent")
RESULTS = ROOT / "outputs" / "plots" / "model_evaluation_results.json"
OUTPUT = ROOT / "outputs" / "plots" / "fig11_per_class_iou_comparison.png"

# Read the actual evaluation results
with open(RESULTS, "r", encoding="utf-8") as f:
    results = json.load(f)

unet_iou = results["unet"]["per_class_iou"]
sub_iou = results["foundation_model"]["per_class_iou"]

classes = ["Urban", "Vegetation", "Water", "Bare Land", "Agriculture"]

# Convert IoU fractions to percentages
unet_values = [unet_iou[c] * 100 for c in classes]
sub_values = [sub_iou[c] * 100 for c in classes]

x = np.arange(len(classes))
width = 0.36

fig, ax = plt.subplots(figsize=(11, 6))

bars1 = ax.bar(x - width / 2, unet_values, width, label="U-Net")
bars2 = ax.bar(
    x + width / 2,
    sub_values,
    width,
    label="ResNet-50 U-Net substitute"
)

ax.set_title("Per-Class IoU Comparison - Guwahati Held-out Evaluation")
ax.set_xlabel("Land-cover class")
ax.set_ylabel("Intersection over Union (IoU, %)")
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=15, ha="right")
ax.set_ylim(0, max(unet_values + sub_values) * 1.18)
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)
ax.legend()

ax.bar_label(bars1, fmt="%.2f", padding=3, fontsize=8)
ax.bar_label(bars2, fmt="%.2f", padding=3, fontsize=8)

fig.tight_layout()
fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close(fig)

print("FIG. 11 GENERATED SUCCESSFULLY")
print(f"Saved to: {OUTPUT}")
print("Values were read from model_evaluation_results.json")
