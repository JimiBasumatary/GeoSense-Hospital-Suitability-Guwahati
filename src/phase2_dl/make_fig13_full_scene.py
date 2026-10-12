
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time
import numpy as np
import torch
import rasterio
import segmentation_models_pytorch as smp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from rasterio.windows import Window

ROOT = os.getcwd()
IMAGE_FILE = os.path.join(
    ROOT, "data", "satellite", "processed",
    "sentinel2_guwahati_2023_processed.tif"
)
LABEL_FILE = os.path.join(
    ROOT, "data", "satellite", "labels", "labels_2023.npy"
)
UNET_FILE = os.path.join(
    ROOT, "models", "saved", "unet_independent_test_best.pth"
)
SUBSTITUTE_FILE = os.path.join(
    ROOT, "models", "saved", "prithvi_finetuned.pth"
)
OUTPUT_DIR = os.path.join(ROOT, "outputs", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIGURE_FILE = os.path.join(
    OUTPUT_DIR, "Fig_13_Guwahati_full_scene_comparison.png"
)
UNET_PRED_FILE = os.path.join(
    ROOT, "outputs", "guwahati_2023_fig13_unet_prediction.tif"
)
SUB_PRED_FILE = os.path.join(
    ROOT, "outputs", "guwahati_2023_fig13_resnet50_substitute_prediction.tif"
)

NUM_CLASSES = 5
CHIP_SIZE = 224
STRIDE = 174
BATCH_SIZE = 4
CLASS_NAMES = ["Urban", "Vegetation", "Water", "Bare Land", "Agriculture"]
COLORS = ["#d73027", "#1a9850", "#4575b4", "#d9c27f", "#fee08b"]
cmap = ListedColormap(COLORS)
cmap.set_bad("black")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path, encoder):
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=6,
        classes=NUM_CLASSES
    )
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print("Loaded:", os.path.basename(path), "| encoder:", encoder)
    return model


def tiled_inference(model, src):
    height, width = src.height, src.width
    rows = list(range(0, max(1, height - CHIP_SIZE + 1), STRIDE))
    cols = list(range(0, max(1, width - CHIP_SIZE + 1), STRIDE))

    if rows[-1] != height - CHIP_SIZE:
        rows.append(max(0, height - CHIP_SIZE))
    if cols[-1] != width - CHIP_SIZE:
        cols.append(max(0, width - CHIP_SIZE))

    windows = [
        (r, c) for r in sorted(set(rows)) for c in sorted(set(cols))
    ]
    probability_sum = np.zeros(
        (NUM_CLASSES, height, width), dtype=np.float32
    )
    count = np.zeros((height, width), dtype=np.uint16)

    print("Tiles to process:", len(windows))
    start_time = time.perf_counter()

    for start in range(0, len(windows), BATCH_SIZE):
        positions = windows[start:start + BATCH_SIZE]
        arrays = []

        for row, col in positions:
            window = Window(
                col, row,
                min(CHIP_SIZE, width - col),
                min(CHIP_SIZE, height - row)
            )
            data = src.read(
                indexes=[1, 2, 3, 4, 5, 6], window=window
            ).astype(np.float32)

            # Match the existing Fig. 10 preprocessing.
            for band in range(data.shape[0]):
                values = data[band]
                finite = np.isfinite(values)
                if finite.any():
                    values[~finite] = values[finite].mean()
                else:
                    values[:] = 0.0

            if data.shape[1:] != (CHIP_SIZE, CHIP_SIZE):
                raise ValueError(
                    f"Unexpected tile shape at {(row, col)}: {data.shape}"
                )
            arrays.append(data)

        batch = torch.from_numpy(np.stack(arrays)).to(device)

        with torch.no_grad():
            probabilities = torch.softmax(model(batch), dim=1).cpu().numpy()

        for k, (row, col) in enumerate(positions):
            probability_sum[
                :, row:row + CHIP_SIZE, col:col + CHIP_SIZE
            ] += probabilities[k]
            count[
                row:row + CHIP_SIZE, col:col + CHIP_SIZE
            ] += 1

        print(f"Processed {min(start + BATCH_SIZE, len(windows))}/{len(windows)}")

    count[count == 0] = 1
    prediction = np.argmax(
        probability_sum / count[np.newaxis, :, :], axis=0
    ).astype(np.uint8)

    print(f"Inference seconds: {time.perf_counter() - start_time:.2f}")
    return prediction


def save_prediction(path, prediction, profile):
    output_profile = profile.copy()
    output_profile.update(
        driver="GTiff", count=1, dtype="uint8",
        nodata=255, compress="deflate"
    )
    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(prediction, 1)
    print("Saved raster:", path)


def main():
    for path in [IMAGE_FILE, LABEL_FILE, UNET_FILE, SUBSTITUTE_FILE]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing required file: {path}")

    labels = np.load(LABEL_FILE)
    unet = load_model(UNET_FILE, "resnet34")
    substitute = load_model(SUBSTITUTE_FILE, "resnet50")

    with rasterio.open(IMAGE_FILE) as src:
        if src.count < 6:
            raise ValueError("Input raster has fewer than six bands.")
        if labels.shape != (src.height, src.width):
            raise ValueError(
                f"Label shape {labels.shape} differs from raster "
                f"shape {(src.height, src.width)}"
            )

        profile = src.profile.copy()
        print("Raster dimensions:", src.height, "x", src.width)
        print("Running U-Net full-scene inference...")
        unet_prediction = tiled_inference(unet, src)

        print("Running ResNet-50 U-Net substitute inference...")
        substitute_prediction = tiled_inference(substitute, src)

    save_prediction(UNET_PRED_FILE, unet_prediction, profile)
    save_prediction(SUB_PRED_FILE, substitute_prediction, profile)

    valid = (
        (labels != 255) & (labels >= 0) & (labels < NUM_CLASSES)
    )
    if not valid.any():
        raise ValueError("No valid reference-label pixels found.")

    unet_match = (unet_prediction == labels) & valid
    substitute_match = (substitute_prediction == labels) & valid
    unet_agreement = 100 * unet_match.sum() / valid.sum()
    substitute_agreement = 100 * substitute_match.sum() / valid.sum()

    print("\nFULL-SCENE AGREEMENT WITH GUWAHATI 2023 PSEUDO-LABELS")
    print("Valid reference pixels:", int(valid.sum()))
    print(f"U-Net agreement: {unet_agreement:.2f}%")
    print(f"ResNet-50 U-Net substitute agreement: {substitute_agreement:.2f}%")

    reference = np.ma.masked_where(~valid, labels)
    pred_unet = np.ma.masked_where(~valid, unet_prediction)
    pred_sub = np.ma.masked_where(~valid, substitute_prediction)

    agreement_unet = np.ma.masked_where(
        ~valid, (unet_prediction == labels).astype(np.uint8)
    )
    agreement_sub = np.ma.masked_where(
        ~valid, (substitute_prediction == labels).astype(np.uint8)
    )
    agreement_cmap = ListedColormap(["#d73027", "#1a9850"])
    agreement_cmap.set_bad("black")

    fig, axes = plt.subplots(1, 5, figsize=(25, 7))
    panels = [
        (reference, cmap, "2023 Rule-Based Pseudo-Labels", 4),
        (pred_unet, cmap, "U-Net Full-Scene Prediction", 4),
        (pred_sub, cmap, "ResNet-50 U-Net Substitute", 4),
        (agreement_unet, agreement_cmap,
         f"U-Net Agreement: {unet_agreement:.2f}%", 1),
        (agreement_sub, agreement_cmap,
         f"Substitute Agreement: {substitute_agreement:.2f}%", 1),
    ]

    for ax, (data, color_map, title, vmax) in zip(axes, panels):
        ax.imshow(data, cmap=color_map, vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()

    legend_items = [
        plt.Line2D(
            [0], [0], marker="s", linestyle="", color="w",
            markerfacecolor=COLORS[i], label=CLASS_NAMES[i], markersize=9
        )
        for i in range(NUM_CLASSES)
    ]
    fig.legend(
        handles=legend_items, loc="lower center",
        ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.02)
    )
    fig.suptitle(
        "Guwahati 2023: Full-Scene Land-Cover Prediction Comparison",
        fontsize=15
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.93], w_pad=0.8)
    fig.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nFIG. 13 COMPLETE")
    print("Figure:", FIGURE_FILE)
    print("Agreement is measured against rule-based pseudo-labels,")
    print("not independently surveyed ground truth.")
    print("The ResNet-50 model is a U-Net substitute, not actual Prithvi weights.")


if __name__ == "__main__":
    main()
