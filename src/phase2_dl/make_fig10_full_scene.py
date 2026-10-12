
import os
import numpy as np
import rasterio
import torch
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from rasterio.windows import Window

# ============================================================
# Fig. 10: Full-scene U-Net prediction comparison
# Study area: Guwahati | Year: 2023
# ============================================================

ROOT = os.getcwd()

IMAGE_FILE = os.path.join(
    ROOT, "data", "satellite", "processed",
    "sentinel2_guwahati_2023_processed.tif"
)

LABEL_FILE = os.path.join(
    ROOT, "data", "satellite", "labels",
    "labels_2023.npy"
)

MODEL_FILE = os.path.join(
    ROOT, "models", "saved",
    "unet_independent_test_best.pth"
)

OUTPUT_DIR = os.path.join(ROOT, "outputs", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PREDICTION_FILE = os.path.join(
    ROOT, "outputs", "guwahati_2023_unet_prediction.tif"
)

FIGURE_FILE = os.path.join(
    OUTPUT_DIR, "Fig_10_Guwahati_full_scene_comparison.png"
)

NUM_CLASSES = 5
CHIP_SIZE = 224
STRIDE = 174
BATCH_SIZE = 4

CLASS_NAMES = [
    "Urban", "Vegetation", "Water",
    "Bare Land", "Agriculture"
]

COLORS = [
    "#d73027", "#1a9850", "#4575b4",
    "#d9c27f", "#fee08b"
]

cmap = ListedColormap(COLORS)
cmap.set_bad("black")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)
print("Loading labels...")

labels = np.load(LABEL_FILE)

# Load trained U-Net
print("Loading trained U-Net...")

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=6,
    classes=NUM_CLASSES
)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=device,
    weights_only=True
)

# Support either a raw state dictionary or a wrapped checkpoint.
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]
elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    checkpoint = checkpoint["model_state_dict"]

model.load_state_dict(checkpoint)
model.to(device)
model.eval()

print("Model loaded successfully.")

# ------------------------------------------------------------
# Full-scene tiled inference
# ------------------------------------------------------------

with rasterio.open(IMAGE_FILE) as src:

    height = src.height
    width = src.width

    if labels.shape != (height, width):
        raise ValueError(
            f"Label shape {labels.shape} does not match "
            f"raster shape {(height, width)}"
        )

    if src.count < 6:
        raise ValueError("The input raster has fewer than 6 bands.")

    print("Raster dimensions:", height, "x", width)
    print("Preparing inference windows...")

    # Include the final edge windows so the entire raster is covered.
    rows = list(range(0, max(1, height - CHIP_SIZE + 1), STRIDE))
    cols = list(range(0, max(1, width - CHIP_SIZE + 1), STRIDE))

    if rows[-1] != height - CHIP_SIZE:
        rows.append(max(0, height - CHIP_SIZE))

    if cols[-1] != width - CHIP_SIZE:
        cols.append(max(0, width - CHIP_SIZE))

    windows = [
        (row, col)
        for row in sorted(set(rows))
        for col in sorted(set(cols))
    ]

    # Accumulate probabilities across overlapping windows.
    probability_sum = np.zeros(
        (NUM_CLASSES, height, width), dtype=np.float32
    )
    prediction_count = np.zeros(
        (height, width), dtype=np.uint16
    )

    total = len(windows)
    print("Total windows:", total)

    for start in range(0, total, BATCH_SIZE):

        batch_positions = windows[start:start + BATCH_SIZE]
        batch_arrays = []

        for row, col in batch_positions:

            window = Window(
                col, row,
                min(CHIP_SIZE, width - col),
                min(CHIP_SIZE, height - row)
            )

            data = src.read(
                indexes=[1, 2, 3, 4, 5, 6],
                window=window
            ).astype(np.float32)

            # Match chip-generation preprocessing:
            # replace non-finite values with each band's
            # mean within this window.
            for band in range(data.shape[0]):
                band_data = data[band]
                valid = np.isfinite(band_data)

                if valid.any():
                    band_mean = band_data[valid].mean()
                    band_data[~valid] = band_mean
                else:
                    band_data[:] = 0.0

            batch_arrays.append(data)

        batch = torch.from_numpy(
            np.stack(batch_arrays)
        ).to(device)

        with torch.no_grad():
            logits = model(batch)
            probabilities = torch.softmax(logits, dim=1)
            probabilities = probabilities.cpu().numpy()

        for k, (row, col) in enumerate(batch_positions):

            tile_h = probabilities.shape[2]
            tile_w = probabilities.shape[3]

            probability_sum[
                :, row:row + tile_h, col:col + tile_w
            ] += probabilities[k]

            prediction_count[
                row:row + tile_h, col:col + tile_w
            ] += 1

        print(
            f"Processed {min(start + BATCH_SIZE, total)}"
            f"/{total} windows"
        )

    # Average probabilities in overlapping regions.
    prediction_count[prediction_count == 0] = 1

    probability_sum /= prediction_count[np.newaxis, :, :]

    prediction = np.argmax(
        probability_sum, axis=0
    ).astype(np.uint8)

    # Save a georeferenced prediction raster.
    profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=255,
        compress="deflate"
    )

    with rasterio.open(PREDICTION_FILE, "w", **profile) as dst:
        dst.write(prediction, 1)

print("Prediction GeoTIFF saved:", PREDICTION_FILE)

# ------------------------------------------------------------
# Fig. 10: Compare reference labels and predictions
# ------------------------------------------------------------

valid = (
    (labels != 255)
    & (labels >= 0)
    & (labels < NUM_CLASSES)
)

agreement = np.full(labels.shape, np.nan, dtype=np.float32)
agreement[valid] = (
    prediction[valid] == labels[valid]
).astype(np.float32)

reference_map = np.ma.masked_where(~valid, labels)
prediction_map = np.ma.masked_where(~valid, prediction)

fig, axes = plt.subplots(
    1, 3, figsize=(19, 7), constrained_layout=True
)

axes[0].imshow(reference_map, cmap=cmap, vmin=0, vmax=4)
axes[0].set_title("2023 Rule-Based Pseudo-Labels")

axes[1].imshow(prediction_map, cmap=cmap, vmin=0, vmax=4)
axes[1].set_title("U-Net Full-Scene Prediction")

agreement_cmap = ListedColormap(["#d73027", "#1a9850"])
agreement_cmap.set_bad("black")

axes[2].imshow(agreement, cmap=agreement_cmap, vmin=0, vmax=1)
axes[2].set_title("Pixel Agreement (Green = Match)")

for ax in axes:
    ax.set_axis_off()

legend_items = [
    plt.Line2D(
        [0], [0], marker="s", linestyle="",
        color="w", markerfacecolor=COLORS[i],
        label=CLASS_NAMES[i], markersize=9
    )
    for i in range(NUM_CLASSES)
]

fig.legend(
    handles=legend_items,
    loc="lower center",
    ncol=5,
    frameon=False
)

fig.suptitle(
    "Guwahati 2023: Full-Scene U-Net Classification Comparison",
    fontsize=15
)

fig.savefig(
    FIGURE_FILE,
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

print("Figure saved:", FIGURE_FILE)

if valid.any():
    overall_agreement = 100 * agreement[valid].mean()
    print(f"Agreement with pseudo-labels: {overall_agreement:.2f}%")
    print("Compared valid pixels:", int(valid.sum()))

print("FIG. 10 COMPLETE")
