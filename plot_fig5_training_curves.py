import json
import matplotlib.pyplot as plt
from pathlib import Path

base = Path("models/evaluation")

with open(base / "prithvi_history.json", "r") as f:
    guwahati = json.load(f)

with open(base / "unet_history.json", "r") as f:
    unet = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Training and validation loss: Guwahati
epochs = range(1, len(guwahati["train_loss"]) + 1)

axes[0].plot(epochs, guwahati["train_loss"], marker="o",
             label="Guwahati training loss")
axes[0].plot(epochs, guwahati["val_loss"], marker="o",
             label="Guwahati validation loss")
axes[0].set_title("Guwahati Fine-Tuning")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Exercise 3 U-Net history
epochs_unet = range(1, len(unet["train_loss"]) + 1)

axes[1].plot(epochs_unet, unet["train_loss"], marker="o",
             label="U-Net training loss")
axes[1].plot(epochs_unet, unet["val_loss"], marker="o",
             label="U-Net validation loss")
axes[1].set_title("Exercise 3 U-Net")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(base / "fig5_training_curves.png", dpi=300)
plt.show()