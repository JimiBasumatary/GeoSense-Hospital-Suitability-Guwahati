
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(os.path.abspath(__file__))

INPUT = os.path.join(
    ROOT, "models", "evaluation",
    "independent_test_results.json"
)

OUTPUT = os.path.join(
    ROOT, "outputs", "plots",
    "unet_independent_test_confusion_matrix_row_normalized.png"
)

CLASS_NAMES = [
    "Urban",
    "Vegetation",
    "Water",
    "Bare Land",
    "Agriculture"
]

with open(INPUT, "r", encoding="utf-8") as f:
    results = json.load(f)

cm = np.asarray(results["confusion_matrix"], dtype=np.int64)

if cm.shape != (5, 5):
    raise ValueError(f"Expected a 5x5 matrix, got {cm.shape}")

row_totals = cm.sum(axis=1, keepdims=True)

cm_percent = np.divide(
    cm * 100.0,
    row_totals,
    out=np.zeros_like(cm, dtype=float),
    where=row_totals != 0
)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

plt.figure(figsize=(10, 8))
ax = sns.heatmap(
    cm_percent,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    vmin=0,
    vmax=100,
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    cbar_kws={"label": "Actual-class pixels (%)"}
)

ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")
ax.set_title("U-Net Independent-Test Confusion Matrix\nGuwahati")
plt.xticks(rotation=30, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close()

print("Independent-test accuracy:",
      results["test_accuracy_percent"], "%")
print("Mean IoU:", results["mean_iou"])
print("Test chips:",
      results["dataset"]["independent_test_chips"])
print("Confusion matrix saved to:")
print(OUTPUT)
