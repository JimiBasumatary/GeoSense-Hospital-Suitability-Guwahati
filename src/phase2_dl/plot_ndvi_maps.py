
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.enums import Resampling

BASE_DIR = r"D:\GeoSense_Agent"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

datasets = [
    ("2015", os.path.join(
        BASE_DIR, "data", "satellite", "processed",
        "sentinel2_guwahati_2015_processed.tif"
    )),
    ("2023", os.path.join(
        BASE_DIR, "data", "satellite", "processed",
        "sentinel2_guwahati_2023_processed.tif"
    ))
]

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

for ax, (year, path) in zip(axes, datasets):
    print(f"Preparing NDVI map for {year}...")

    with rasterio.open(path) as src:
        # Reduce resolution for plotting; original rasters remain unchanged
        height = min(src.height, 1200)
        width = min(src.width, 1200)

        ndvi = src.read(
            7,
            out_shape=(height, width),
            masked=True,
            resampling=Resampling.average
        ).astype(np.float32)

        bounds = src.bounds

    # Mask invalid pixels and values outside the NDVI range
    values = np.asarray(ndvi.filled(np.nan))
    values[(values < -1) | (values > 1)] = np.nan

    im = ax.imshow(
        values,
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        extent=[
            bounds.left, bounds.right,
            bounds.bottom, bounds.top
        ],
        origin="upper",
        interpolation="nearest"
    )

    ax.set_title(f"Guwahati NDVI — {year}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.ticklabel_format(useOffset=False, style="plain")
    print(f"Map prepared for {year}")

fig.subplots_adjust(right=0.87, wspace=0.25)
cbar_ax = fig.add_axes([0.89, 0.17, 0.025, 0.66])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("NDVI (−1 to +1)")

fig.suptitle(
    "Spatial Distribution of NDVI in Guwahati (2015 and 2023)",
    fontsize=13
)
fig.savefig(
    os.path.join(
        OUTPUT_DIR, "Fig_10_Guwahati_NDVI_Maps_2015_2023.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

print("FIG. 10 SAVED:", os.path.join(
    OUTPUT_DIR, "Fig_10_Guwahati_NDVI_Maps_2015_2023.png"
))
