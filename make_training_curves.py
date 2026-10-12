
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = r"D:\GeoSense_Agent"

history_path = os.path.join(
    root, "models", "evaluation", "unet_history.json"
)

output_path = os.path.join(
    root, "outputs", "fig7_guwahati_training_curves.png"
)

with open(history_path, "r", encoding="utf-8") as f:
    history = json.load(f)

epochs = range(1, len(history["train_loss"]) + 1)

# Create separate plots for loss and accuracy
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Training and validation loss
axes[0].plot(
    epochs, history["train_loss"],
    marker="o", label="Training Loss"
)
axes[0].plot(
    epochs, history["val_loss"],
    marker="s", label="Validation Loss"
)
axes[0].set_title("U-Net Training and Validation Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_xticks(list(epochs))
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Validation accuracy
axes[1].plot(
    epochs, history["val_accuracy"],
    marker="o", label="Validation Accuracy"
)
axes[1].set_title("U-Net Validation Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_xticks(list(epochs))
axes[1].set_ylim(0, 100)
axes[1].grid(True, alpha=0.3)
axes[1].legend()

fig.suptitle(
    "Guwahati U-Net Training History (5 Epochs)"
)
plt.tight_layout()

os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print("Training curves generated successfully:")
print(output_path)
