from pathlib import Path
import random
import numpy as np
import rasterio
import matplotlib.pyplot as plt

from rasterio.enums import Resampling

roots = [
    Path(r"data\satellite\chips\2023"),
    Path(r"data\satellite\colab_dataset\chips"),
    Path(r"data\satellite\colab_small\chips"),
]

files = sorted({
    f for root in roots if root.exists()
    for pattern in ("*.tif", "*.tiff")
    for f in root.rglob(pattern)
})

print("TIFF chips found:", len(files))

if not files:
    raise SystemExit("No TIFF chips found. Check the chip folders.")

random.seed(42)
selected = random.sample(files, min(12, len(files)))

fig, axes = plt.subplots(3, 4, figsize=(15, 10))
axes = axes.flatten()
shown = 0

for file in selected:
    try:
        with rasterio.open(file) as src:
            if src.count < 3:
                continue

            bands = [3, 2, 1] if src.count >= 4 else [1, 2, 3]
            image = src.read(
                bands,
                out_shape=(3, 224, 224),
                resampling=Resampling.bilinear
            ).astype(np.float32)
            nodata = src.nodata

        image = np.moveaxis(image, 0, -1)

        for band in range(3):
            values = image[:, :, band]
            valid = np.isfinite(values)

            if nodata is not None:
                valid &= values != nodata

            if valid.any():
                low, high = np.percentile(values[valid], [2, 98])
                image[:, :, band] = np.clip(
                    (values - low) / (high - low + 1e-8), 0, 1
                )

        axes[shown].imshow(np.nan_to_num(image))
        axes[shown].set_title(file.name, fontsize=8)
        axes[shown].axis("off")
        shown += 1

        if shown == 12:
            break

    except Exception as error:
        print("Skipped:", file.name, error)

for ax in axes[shown:]:
    ax.axis("off")

fig.suptitle(
    "Guwahati Training Chips (2023) - True Colour B4/B3/B2",
    fontsize=14
)
plt.tight_layout()

output = Path(
    r"data\satellite\processed\guwahati_training_chip_montage_2023.png"
)
output.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output, dpi=200, bbox_inches="tight")

print("Chips displayed:", shown)
print("Montage saved:", output.resolve())
plt.show()