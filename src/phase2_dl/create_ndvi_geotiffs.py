
import os
import numpy as np
import rasterio

BASE = r"D:\GeoSense_Agent"
INPUT_DIR = os.path.join(BASE, "data", "satellite", "processed")
OUTPUT_DIR = os.path.join(BASE, "data", "satellite", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

datasets = {
    "2015": os.path.join(
        INPUT_DIR, "sentinel2_guwahati_2015_processed.tif"
    ),
    "2023": os.path.join(
        INPUT_DIR, "sentinel2_guwahati_2023_processed.tif"
    )
}

for year, input_path in datasets.items():
    output_path = os.path.join(
        OUTPUT_DIR, f"ndvi_guwahati_{year}.tif"
    )

    print(f"\nCreating NDVI GeoTIFF for {year}...")

    with rasterio.open(input_path) as src:
        # Spectral band order: B2, B3, B4, B8, B11, B12
        red = src.read(3).astype("float32")
        nir = src.read(4).astype("float32")

        profile = src.profile.copy()

        # Calculate NDVI safely, avoiding division by zero
        denominator = nir + red
        valid = (
            np.isfinite(red)
            & np.isfinite(nir)
            & (denominator != 0)
        )

        ndvi = np.full(red.shape, np.nan, dtype="float32")
        ndvi[valid] = (
            (nir[valid] - red[valid]) / denominator[valid]
        )

        # Mask impossible values and invalid input pixels
        ndvi[(ndvi < -1) | (ndvi > 1)] = np.nan

        profile.update(
            count=1,
            dtype="float32",
            nodata=-9999.0,
            compress="deflate"
        )

        output = np.where(
            np.isfinite(ndvi), ndvi, -9999.0
        ).astype("float32")

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(output, 1)
            dst.set_band_description(1, f"NDVI_{year}")

    print("Saved:", output_path)
    print("Valid NDVI pixels:", int(np.isfinite(ndvi).sum()))
    print("NDVI minimum:", float(np.nanmin(ndvi)))
    print("NDVI maximum:", float(np.nanmax(ndvi)))
    print("NDVI mean:", float(np.nanmean(ndvi)))

print("\nNDVI GeoTIFF generation finished.")
