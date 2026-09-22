import os
import json
import numpy as np
import rasterio
from rasterio.windows import Window

# ============================================================
# GeoSense Agent 2.0 - Phase 2
# Exercise 2: Sentinel-2 Image Chipping
# Study Area: Guwahati
# Year: 2023
# ============================================================

input_file = (
    r"data\satellite\processed"
    r"\sentinel2_guwahati_2023_processed.tif"
)

output_dir = r"data\satellite\chips\2023"

os.makedirs(output_dir, exist_ok=True)

chip_size = 224
stride = 174
max_nan_fraction = 0.20

print("=" * 70)
print("GeoSense Phase 2 - Sentinel-2 Chip Generation")
print("=" * 70)

print()
print("Input:", input_file)
print("Output:", output_dir)
print("Chip size:", chip_size, "x", chip_size)
print("Stride:", stride)
print("Overlap:", chip_size - stride, "pixels")
print("Maximum allowed NaN:", max_nan_fraction * 100, "%")

total_windows = 0
saved_chips = 0
skipped_chips = 0

with rasterio.open(input_file) as src:

    print()
    print("Image size:", src.width, "x", src.height)
    print("Bands:", src.count)
    print("CRS:", src.crs)

    for row in range(0, src.height - chip_size + 1, stride):

        for col in range(
            0,
            src.width - chip_size + 1,
            stride
        ):

            total_windows += 1

            window = Window(
                col_off=col,
                row_off=row,
                width=chip_size,
                height=chip_size
            )

            data = src.read(
                window=window
            ).astype(np.float32)

            # ------------------------------------------------
            # Check NaN percentage
            # ------------------------------------------------

            nan_fraction = np.isnan(data).mean()

            if nan_fraction > max_nan_fraction:

                skipped_chips += 1
                continue

            # ------------------------------------------------
            # Fill remaining NaN values using band mean
            # ------------------------------------------------

            for band in range(data.shape[0]):

                band_data = data[band]

                nan_mask = np.isnan(band_data)

                if np.any(nan_mask):

                    valid_values = band_data[~nan_mask]

                    if valid_values.size > 0:

                        band_mean = np.mean(valid_values)

                        band_data[nan_mask] = band_mean

                    else:

                        band_data[nan_mask] = 0

            # ------------------------------------------------
            # Save chip
            # ------------------------------------------------

            chip_name = (
                f"guwahati_2023_"
                f"row_{row:05d}_"
                f"col_{col:05d}.tif"
            )

            chip_path = os.path.join(
                output_dir,
                chip_name
            )

            profile = src.profile.copy()

            profile.update(
                width=chip_size,
                height=chip_size,
                count=src.count,
                dtype="float32",
                driver="GTiff",
                compress="deflate",
                transform=rasterio.windows.transform(
                    window,
                    src.transform
                )
            )

            with rasterio.open(
                chip_path,
                "w",
                **profile
            ) as dst:

                dst.write(data)

            saved_chips += 1

            if saved_chips % 100 == 0:

                print(
                    "Saved chips:",
                    saved_chips
                )

# ------------------------------------------------------------
# Metadata
# ------------------------------------------------------------

metadata = {
    "study_area": "Guwahati",
    "year": 2023,
    "input_file": input_file,
    "chip_size_pixels": chip_size,
    "stride_pixels": stride,
    "overlap_pixels": chip_size - stride,
    "maximum_nan_fraction": max_nan_fraction,
    "total_windows": total_windows,
    "saved_chips": saved_chips,
    "skipped_chips": skipped_chips,
    "bands": [
        "B2_Blue",
        "B3_Green",
        "B4_Red",
        "B8_NIR",
        "B11_SWIR1",
        "B12_SWIR2",
        "NDVI",
        "NDWI"
    ]
}

metadata_file = os.path.join(
    output_dir,
    "metadata_2023.json"
)

with open(
    metadata_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )

print()
print("=" * 70)
print("CHIP GENERATION COMPLETE")
print("=" * 70)

print("Total windows:", total_windows)
print("Saved chips:", saved_chips)
print("Skipped chips:", skipped_chips)
print("Metadata:", metadata_file)