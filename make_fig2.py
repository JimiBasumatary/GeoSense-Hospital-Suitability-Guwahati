
import os
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

root = r"D:\GeoSense_Agent"

image_path = os.path.join(
    root, "data", "satellite", "processed",
    "sentinel2_guwahati_2023_processed.tif"
)
label_path = os.path.join(
    root, "data", "satellite", "labels", "labels_2023.npy"
)
output_path = os.path.join(
    root, "outputs", "fig2_guwahati_2023_composite_labels.png"
)

os.makedirs(os.path.dirname(output_path), exist_ok=True)

print("Loading satellite image...")
with rasterio.open(image_path) as src:
    blue = src.read(1).astype(np.float32)
    green = src.read(2).astype(np.float32)
    red = src.read(3).astype(np.float32)
    print("Image dimensions:", src.height, "x", src.width)

print("Loading pseudo-label array...")
labels = np.load(label_path)
print("Label dimensions:", labels.shape)

if labels.shape != red.shape:
    raise ValueError("Image and label dimensions do not match.")

def stretch(band):
    valid = np.isfinite(band) & (band != 0)
    if not valid.any():
        return np.zeros_like(band)
    low, high = np.percentile(band[valid], [2, 98])
    if high <= low:
        return np.zeros_like(band)
    result = np.clip((band - low) / (high - low), 0, 1)
    result[~np.isfinite(result)] = 0
    return result

rgb = np.dstack([stretch(red), stretch(green), stretch(blue)])

colors = ["#d73027", "#1a9850", "#4575b4", "#bdbdbd", "#fee08b"]
names = ["Urban", "Vegetation", "Water", "Bare Land", "Agriculture"]

cmap = ListedColormap(colors)
cmap.set_bad("black")
norm = BoundaryNorm(np.arange(-0.5, 5.5, 1), cmap.N)

valid = np.isin(labels, [0, 1, 2, 3, 4])
label_map = np.ma.masked_where(~valid, labels)

print("Creating figure...")
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].imshow(rgb)
axes[0].set_title("Guwahati Sentinel-2 Composite (2023)")
axes[0].axis("off")

axes[1].imshow(label_map, cmap=cmap, norm=norm, interpolation="nearest")
axes[1].set_title("Guwahati Rule-Based Pseudo-Labels (2023)")
axes[1].axis("off")

handles = [
    plt.Line2D(
        [0], [0], marker="s", linestyle="",
        markerfacecolor=colors[i], markeredgecolor="black",
        markersize=9, label=names[i]
    )
    for i in range(5)
]

fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
fig.suptitle("Guwahati: Sentinel-2 Composite and Pseudo-Labels, 2023")
plt.tight_layout(rect=[0, 0.10, 1, 0.93])

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print("Figure saved successfully:")
print(output_path)
