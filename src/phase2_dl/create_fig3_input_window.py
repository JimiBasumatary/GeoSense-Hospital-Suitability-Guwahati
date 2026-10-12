
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

RASTER = (
    r"data\satellite\processed"
    r"\sentinel2_guwahati_2023_processed.tif"
)

OUTPUT = r"outputs\plots\fig3_model_input_window.png"

# Original Fig. 1 test coordinate
LAT = 26.1445
LON = 91.7362
CHIP_SIZE = 224

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

with rasterio.open(RASTER) as src:
    if not (src.bounds.left <= LON <= src.bounds.right
            and src.bounds.bottom <= LAT <= src.bounds.top):
        raise ValueError("Selected coordinate is outside the raster.")

    row, col = src.index(LON, LAT)

    # Match the 224 x 224 extraction in image_classifier.py
    half = CHIP_SIZE // 2
    row_start = max(0, row - half)
    col_start = max(0, col - half)
    row_end = min(src.height, row_start + CHIP_SIZE)
    col_end = min(src.width, col_start + CHIP_SIZE)

    # Shift the window inward at raster edges
    row_start = max(0, row_end - CHIP_SIZE)
    col_start = max(0, col_end - CHIP_SIZE)

    window = rasterio.windows.Window(
        col_start, row_start,
        col_end - col_start,
        row_end - row_start
    )

    # Bands 3, 2, 1 = Red, Green, Blue for the RGB display
    rgb = src.read([3, 2, 1]).astype(np.float32)
    patch = src.read(
        [3, 2, 1], window=window
    ).astype(np.float32)

    # Geographic extent of the actual model-input window
    transform = rasterio.windows.transform(window, src.transform)
    left = transform.c
    top = transform.f
    right = left + window.width * transform.a
    bottom = top + window.height * transform.e

    bounds = src.bounds

def stretch(image):
    result = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        band = image[i]
        valid = np.isfinite(band)
        if valid.any():
            low, high = np.percentile(band[valid], [2, 98])
            if high > low:
                result[i] = np.clip(
                    (np.nan_to_num(band, nan=low) - low)
                    / (high - low), 0, 1
                )
    return np.moveaxis(result, 0, -1)

full_rgb = stretch(rgb)
patch_rgb = stretch(patch)

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# Full scene
axes[0].imshow(
    full_rgb,
    extent=[bounds.left, bounds.right,
            bounds.bottom, bounds.top],
    origin="upper"
)
axes[0].add_patch(Rectangle(
    (left, bottom), right - left, top - bottom,
    fill=False, edgecolor="yellow", linewidth=2
))
axes[0].scatter(
    LON, LAT, marker="+", s=100,
    color="red", linewidths=2
)
axes[0].set_title("Guwahati 2023 satellite image")
axes[0].set_xlabel("Longitude (degrees)")
axes[0].set_ylabel("Latitude (degrees)")
axes[0].text(
    0.03, 0.04, "Yellow: model-input window\nRed +: selected coordinate",
    transform=axes[0].transAxes,
    color="white", fontsize=9,
    bbox=dict(facecolor="black", alpha=0.7)
)

# Actual extracted patch, displayed as RGB
axes[1].imshow(patch_rgb, origin="upper")
axes[1].scatter(
    col - col_start, row - row_start,
    marker="+", s=100, color="red", linewidths=2
)
axes[1].set_title(
    f"224 x 224 pixel input window\n"
    f"Coordinate: {LAT}, {LON}"
)
axes[1].set_xlabel("Pixel column")
axes[1].set_ylabel("Pixel row")

fig.suptitle(
    "GeoSense Agent — Fig. 3: Location and Model Input Window",
    fontsize=14
)
plt.tight_layout()
plt.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close()

print("Fig. 3 saved to:", OUTPUT)
print("Selected pixel (row, column):", row, col)
print("Window dimensions:", window.width, "x", window.height)
print("Window bounds:", left, bottom, right, top)
