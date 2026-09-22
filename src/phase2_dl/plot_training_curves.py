import os
import json
import matplotlib.pyplot as plt


# ============================================================
# GeoSense Phase 2 - Exercise 3.2
# U-Net Training Curves
# Study Area: Guwahati
# ============================================================

HISTORY_FILE = r"models\evaluation\unet_history.json"
OUTPUT_DIR = r"models\evaluation"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "unet_training_curves.png"
)


# ------------------------------------------------------------
# Load training history
# ------------------------------------------------------------

with open(HISTORY_FILE, "r") as f:
    history = json.load(f)


epochs = range(
    1,
    len(history["train_loss"]) + 1
)


# ============================================================
# Figure 1: Training and Validation Loss
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    history["train_loss"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    history["val_loss"],
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(
    "U-Net Training and Validation Loss - Guwahati"
)

plt.xticks(list(epochs))
plt.legend()
plt.grid(True, alpha=0.3)

loss_file = os.path.join(
    OUTPUT_DIR,
    "unet_loss_curve.png"
)

plt.tight_layout()
plt.savefig(
    loss_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# Figure 2: Validation Accuracy
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    history["val_accuracy"],
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title(
    "U-Net Validation Accuracy - Guwahati"
)

plt.xticks(list(epochs))
plt.legend()
plt.grid(True, alpha=0.3)

accuracy_file = os.path.join(
    OUTPUT_DIR,
    "unet_validation_accuracy.png"
)

plt.tight_layout()
plt.savefig(
    accuracy_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("TRAINING CURVES CREATED")
print("=" * 70)

print()
print("Loss curve:")
print(loss_file)

print()
print("Validation accuracy:")
print(accuracy_file)

print()
print("Final validation accuracy:")
print(
    f"{history['val_accuracy'][-1]:.2f}%"
)

print()
print("=" * 70)