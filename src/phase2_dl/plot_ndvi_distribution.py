
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# Project paths
BASE_DIR = r"D:\GeoSense_Agent"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

datasets = {
    "2015": os.path.join(
        BASE_DIR, "data", "satellite", "processed",
        "sentinel2_guwahati_2015_processed.tif"
    ),
    "2023": os.path.join(
        BASE_DIR, "data", "satellite", "processed",
        "sentinel2_guwahati_2023_processed.tif"
    )
}

ndvi_data = {}

# Read NDVI from Band 7 of each processed raster
for year, path in datasets.items():
    print(f"\nReading Guwahati NDVI for {year}...")

    with rasterio.open(path) as src:
        if src.count < 7:
            raise ValueError(f"{year} raster has no Band 7.")

        ndvi = src.read(7, masked=True).astype(np.float32)
        values = np.asarray(ndvi.compressed())
        values = values[np.isfinite(values)]
        values = values[(values >= -1) & (values <= 1)]

    if values.size == 0:
        raise ValueError(f"No valid NDVI values found for {year}.")

    ndvi_data[year] = values

    print(f"Valid NDVI pixels: {values.size:,}")
    print(f"Minimum NDVI: {values.min():.4f}")
    print(f"Maximum NDVI: {values.max():.4f}")
    print(f"Mean NDVI: {values.mean():.4f}")

# Fig. 9: Compare NDVI distributions
fig, ax = plt.subplots(figsize=(10, 6))

for year, values in ndvi_data.items():
    # Deterministic sample for efficient plotting
    if values.size > 1_000_000:
        rng = np.random.default_rng(42)
        values = rng.choice(values, 1_000_000, replace=False)

    ax.hist(
        values,
        bins=100,
        range=(-1, 1),
        density=True,
        histtype="step",
        linewidth=1.8,
        label=year
    )

ax.set_title("NDVI Distribution in Guwahati (2015 and 2023)")
ax.set_xlabel("NDVI")
ax.set_ylabel("Probability Density")
ax.set_xlim(-1, 1)
ax.grid(True, alpha=0.25)
ax.legend(title="Year")
fig.tight_layout()

output_path = os.path.join(
    OUTPUT_DIR, "Fig_9_Guwahati_NDVI_Distribution_2015_2023.png"
)
fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("\nFIG. 9 SAVED:", output_path)
