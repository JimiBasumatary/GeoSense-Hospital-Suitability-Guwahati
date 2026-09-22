import os
import json
import matplotlib.pyplot as plt

# ============================================================
# GeoSense Phase 2 - Exercise 4.2
# Training Curve Comparison
# ============================================================

UNET_HISTORY = r"models\evaluation\unet_history.json"
FOUNDATION_HISTORY = r"models\evaluation\prithvi_history.json"
OUTPUT_DIR = r"outputs\plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "training_curve_comparison.png"
)

print("=" * 70)
print("GeoSense Phase 2 - Training Curve Comparison")
print("=" * 70)

# ------------------------------------------------------------
# Load U-Net history
# ------------------------------------------------------------

with open(UNET_HISTORY, "r") as f:
    unet_history = json.load(f)

# ------------------------------------------------------------
# Load Foundation-model history
# ------------------------------------------------------------

with open(FOUNDATION_HISTORY, "r") as f:
    foundation_history = json.load(f)

# ------------------------------------------------------------
# Extract values
# ------------------------------------------------------------

unet_train_loss = unet_history["train_loss"]
unet_val_loss = unet_history["val_loss"]

foundation_train_loss = foundation_history["train_loss"]
foundation_val_loss = foundation_history["val_loss"]

unet_epochs = range(1, len(unet_train_loss) + 1)
foundation_epochs = range(1, len(foundation_train_loss) + 1)

# ------------------------------------------------------------
# Create comparison figure
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    unet_epochs,
    unet_train_loss,
    marker="o",
    label="U-Net Training Loss"
)

plt.plot(
    unet_epochs,
    unet_val_loss,
    marker="o",
    label="U-Net Validation Loss"
)

plt.plot(
    foundation_epochs,
    foundation_train_loss,
    marker="s",
    label="Foundation Model Training Loss"
)

plt.plot(
    foundation_epochs,
    foundation_val_loss,
    marker="s",
    label="Foundation Model Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(
    "Training and Validation Loss Comparison\n"
    "Guwahati Sentinel-2 Semantic Segmentation"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("Training curve comparison created successfully.")
print("Output:", OUTPUT_FILE)
print()
print("=" * 70)
print("COMPLETE")
print("=" * 70)