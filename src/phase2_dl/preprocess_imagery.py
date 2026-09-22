import os
import numpy as np
import rasterio
from rasterio.windows import Window

# ============================================================
# GeoSense Agent 2.0 - Phase 2
# Exercise 2: Sentinel-2 Preprocessing
# Study Area: Guwahati
# Year: 2023
# ============================================================

input_file = r"data\satellite\raw\sentinel2_guwahati_2023_mosaic.tif"

output_dir = r"data\satellite\processed"

os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(
    output_dir,
    "sentinel2_guwahati_2023_processed.tif"
)

print("=" * 70)
print("GeoSense Phase 2 - Sentinel-2 Preprocessing")
print("=" * 70)

print()
print("Input:", input_file)
print("Output:", output_file)

with rasterio.open(input_file) as src:

    print()
    print("Input CRS:", src.crs)
    print("Input size:", src.width, "x", src.height)
    print("Input bands:", src.count)
    print("Input resolution:", src.res)

    profile = src.profile.copy()

    profile.update(
        dtype="float32",
        count=8,
        driver="GTiff",
        compress="deflate",
        nodata=np.nan
    )

    with rasterio.open(output_file, "w", **profile) as dst:

        # Process 100 rows at a time
        rows_per_chunk = 100

        total_chunks = (
            src.height + rows_per_chunk - 1
        ) // rows_per_chunk

        for chunk_number, row_start in enumerate(
            range(0, src.height, rows_per_chunk),
            start=1
        ):

            row_height = min(
                rows_per_chunk,
                src.height - row_start
            )

            window = Window(
                col_off=0,
                row_off=row_start,
                width=src.width,
                height=row_height
            )

            data = src.read(
                indexes=[1, 2, 3, 4, 5, 6],
                window=window
            ).astype(np.float32)

            # ------------------------------------------------
            # Mask reflectance values above 3000
            # ------------------------------------------------

            data[data > 3000] = np.nan

            # ------------------------------------------------
            # Normalize Sentinel-2 reflectance
            # ------------------------------------------------

            data = data / 10000.0

            # Keep values between 0 and 1
            data = np.clip(data, 0, 1)

            # Band definitions:
            # 1 = B2 Blue
            # 2 = B3 Green
            # 3 = B4 Red
            # 4 = B8 NIR
            # 5 = B11 SWIR1
            # 6 = B12 SWIR2

            green = data[1]
            red = data[2]
            nir = data[3]

            # ------------------------------------------------
            # NDVI
            # NDVI = (NIR - Red) / (NIR + Red)
            # ------------------------------------------------

            ndvi_denominator = nir + red

            ndvi = np.divide(
                nir - red,
                ndvi_denominator,
                out=np.full_like(nir, np.nan),
                where=ndvi_denominator != 0
            )

            # ------------------------------------------------
            # NDWI
            # NDWI = (Green - NIR) / (Green + NIR)
            # ------------------------------------------------

            ndwi_denominator = green + nir

            ndwi = np.divide(
                green - nir,
                ndwi_denominator,
                out=np.full_like(green, np.nan),
                where=ndwi_denominator != 0
            )

            # ------------------------------------------------
            # Create 8-band output
            #
            # Band 1 = B2
            # Band 2 = B3
            # Band 3 = B4
            # Band 4 = B8
            # Band 5 = B11
            # Band 6 = B12
            # Band 7 = NDVI
            # Band 8 = NDWI
            # ------------------------------------------------

            output_data = np.stack(
                [
                    data[0],
                    data[1],
                    data[2],
                    data[3],
                    data[4],
                    data[5],
                    ndvi,
                    ndwi
                ]
            )

            dst.write(
                output_data,
                window=window
            )

            if chunk_number % 5 == 0 or chunk_number == total_chunks:
                print(
                    f"Processed chunks: "
                    f"{chunk_number}/{total_chunks}"
                )

print()
print("=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)
print("Output:", output_file)