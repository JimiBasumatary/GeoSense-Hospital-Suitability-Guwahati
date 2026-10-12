
import os
import json
import re
import numpy as np
import rasterio
import torch
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.abspath(__file__))
CHIP_DIR = os.path.join(ROOT, "data", "satellite", "colab_small", "chips", "2023")
LABEL_FILE = os.path.join(ROOT, "data", "satellite", "labels", "labels_2023.npy")
RESULTS_FILE = os.path.join(ROOT, "models", "evaluation", "independent_test_results.json")
CHECKPOINT = os.path.join(ROOT, "models", "saved", "unet_independent_test_best.pth")
OUTPUT = os.path.join(ROOT, "outputs", "plots", "Fig_9_Guwahati_validation_chips.png")

CLASS_NAMES = ["Urban", "Vegetation", "Water", "Bare Land", "Agriculture"]
COLORS = ["#e31a1c", "#238b45", "#2171b5", "#d9b44a", "#8c6d31"]
NODATA = 255
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rgb_stretch(chip):
    # Display bands 1, 2, 3 as RGB with a percentile stretch.
    rgb = np.transpose(chip[:3], (1, 2, 0)).astype(np.float32)
    rgb = np.nan_to_num(rgb, nan=0, posinf=0, neginf=0)
    out = np.zeros_like(rgb, dtype=np.float32)
    for b in range(3):
        valid = rgb[:, :, b][np.isfinite(rgb[:, :, b])]
        if valid.size:
            lo, hi = np.percentile(valid, [2, 98])
            if hi > lo:
                out[:, :, b] = np.clip((rgb[:, :, b] - lo) / (hi - lo), 0, 1)
    return out

def read_chip(path):
    with rasterio.open(path) as src:
        chip = src.read(indexes=[1, 2, 3, 4, 5, 6]).astype(np.float32)
    match = re.search(r"row_(\d+)_col_(\d+)", os.path.basename(path))
    if not match:
        raise ValueError(f"Cannot parse chip coordinates: {path}")
    row, col = map(int, match.groups())
    label_full = np.load(LABEL_FILE, mmap_mode="r")
    label = np.asarray(label_full[row:row + chip.shape[1], col:col + chip.shape[2]])
    if label.shape != chip.shape[1:]:
        raise ValueError(f"Label/chip shape mismatch for {os.path.basename(path)}")
    chip = np.nan_to_num(chip, nan=0.0, posinf=1.0, neginf=0.0)
    return chip, label

def main():
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Use the validation indices saved by the independent-test experiment.
    validation_indices = results["split_indices"]["validation"]
    chips = sorted(
        f for f in os.listdir(CHIP_DIR) if f.lower().endswith(".tif")
    )
    if len(chips) != results["dataset"]["total_chips"]:
        raise ValueError("Current chip count does not match saved split metadata.")

    selected = validation_indices[:4]
    if len(selected) < 4:
        raise ValueError("Fewer than four validation chips are available.")

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=6,
        classes=5
    ).to(DEVICE)

    state = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    cmap = ListedColormap(COLORS)
    fig, axes = plt.subplots(4, 4, figsize=(17, 16))
    headings = ["True-colour RGB", "Pseudo-label mask",
                "U-Net prediction", "Pixel agreement"]
    for j, heading in enumerate(headings):
        axes[0, j].set_title(heading, fontsize=12, fontweight="bold")

    legend_items = [Patch(facecolor=COLORS[i], label=CLASS_NAMES[i])
                    for i in range(5)]
    legend_items.append(Patch(facecolor="#000000", label="NoData / invalid"))

    for row_plot, idx in enumerate(selected):
        chip_path = os.path.join(CHIP_DIR, chips[idx])
        chip, label = read_chip(chip_path)

        x = torch.from_numpy(chip).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(x)
            prediction = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        valid = (
            (label != NODATA) &
            (label >= 0) & (label < len(CLASS_NAMES))
        )
        agreement = np.full(label.shape, NODATA, dtype=np.uint8)
        agreement[valid] = (prediction[valid] == label[valid]).astype(np.uint8)

        axes[row_plot, 0].imshow(rgb_stretch(chip))
        axes[row_plot, 1].imshow(
            np.ma.masked_where(~valid, label),
            cmap=cmap, vmin=0, vmax=4, interpolation="nearest"
        )
        axes[row_plot, 2].imshow(
            np.ma.masked_where(~valid, prediction),
            cmap=cmap, vmin=0, vmax=4, interpolation="nearest"
        )

        # Agreement: green = correct, red = incorrect, black = invalid.
        agree_cmap = ListedColormap(["#d73027", "#1a9850"])
        agree_masked = np.ma.masked_where(~valid, agreement)
        axes[row_plot, 3].imshow(agree_masked, cmap=agree_cmap,
                                 vmin=0, vmax=1, interpolation="nearest")
        axes[row_plot, 3].set_facecolor("black")

        if valid.any():
            accuracy = 100 * agreement[valid].mean()
            label_text = f"{os.path.basename(chip_path)}\nValid-pixel agreement: {accuracy:.1f}%"
        else:
            label_text = f"{os.path.basename(chip_path)}\nNo valid labelled pixels"

        axes[row_plot, 0].set_ylabel(label_text, fontsize=8)
        for ax in axes[row_plot]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.legend(handles=legend_items, loc="lower center",
               ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.015))
    fig.suptitle("Guwahati 2023: Validation Chip Predictions and Pixel Agreement",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    plt.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved Fig. 9:", OUTPUT)
    print("Validation chip indices:", selected)
    print("Checkpoint:", CHECKPOINT)

if __name__ == "__main__":
    main()
