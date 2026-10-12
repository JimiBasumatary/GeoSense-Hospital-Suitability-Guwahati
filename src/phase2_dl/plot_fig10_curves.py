
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(r"D:\GeoSense_Agent")
EVAL_DIR = ROOT / "models" / "evaluation"
OUT_DIR = ROOT / "outputs" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the actual saved training histories
with open(EVAL_DIR / "unet_history.json", "r") as f:
    unet = json.load(f)

with open(EVAL_DIR / "prithvi_history.json", "r") as f:
    substitute = json.load(f)

epochs_unet = range(1, len(unet["val_loss"]) + 1)
epochs_sub = range(1, len(substitute["val_loss"]) + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel 1: Validation loss
axes[0].plot(
    epochs_unet, unet["val_loss"],
    marker="o", label="U-Net"
)
axes[0].plot(
    epochs_sub, substitute["val_loss"],
    marker="s", label="ResNet-50 U-Net substitute"
)
axes[0].set_yscale("log")
axes[0].set_title("Validation Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Validation loss (log scale)")
axes[0].set_xticks(range(1, 6))
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Panel 2: Validation pixel accuracy
axes[1].plot(
    epochs_unet, unet["val_accuracy"],
    marker="o", label="U-Net"
)
axes[1].plot(
    epochs_sub, substitute["val_accuracy"],
    marker="s", label="ResNet-50 U-Net substitute"
)
axes[1].set_title("Validation Pixel Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Pixel accuracy (%)")
axes[1].set_xticks(range(1, 6))
axes[1].grid(True, alpha=0.3)
axes[1].legend()

fig.suptitle(
    "Guwahati Segmentation Models: Validation Curves",
    fontsize=14
)
fig.tight_layout()

output = OUT_DIR / "fig10_validation_loss_accuracy_comparison.png"
fig.savefig(output, dpi=300, bbox_inches="tight")
plt.close(fig)

print("FIG. 10 GENERATED SUCCESSFULLY")
print(f"Saved to: {output}")
print("Source: saved U-Net and substitute-model history JSON files")
print("Note: Validation loss uses a logarithmic axis.")
