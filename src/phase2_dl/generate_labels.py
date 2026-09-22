import os
import numpy as np
import rasterio


# ============================================================
# GeoSense Phase 2 - Exercise 3.1
# Generate Ground-Truth Labels from Spectral Rules
# Study Area: Guwahati
# ============================================================

INPUT_DIR = r"data\satellite\processed"
OUTPUT_DIR = r"data\satellite\labels"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# Class definitions
# ------------------------------------------------------------
# 0 = Urban
# 1 = Vegetation
# 2 = Water
# 3 = Bare Land
# 4 = Agriculture
# 255 = NoData / Ignore
# ------------------------------------------------------------

CLASS_NAMES = {
    0: "Urban",
    1: "Vegetation",
    2: "Water",
    3: "Bare Land",
    4: "Agriculture",
    255: "NoData"
}


def classify_pixels(ndvi, ndwi, nir, swir):
    """
    Apply the spectral threshold rules from Exercise 3.1.
    """

    labels = np.full(ndvi.shape, 3, dtype=np.uint8)

    # NoData mask
    valid = (
        np.isfinite(ndvi)
        & np.isfinite(ndwi)
        & np.isfinite(nir)
        & np.isfinite(swir)
    )

    # Default: Bare Land
    labels[:] = 3

    # Water
    water = valid & (ndwi > 0.3)
    labels[water] = 2

    # Vegetation
    vegetation = valid & (ndvi > 0.4)
    labels[vegetation] = 1

    # Agriculture
    agriculture = valid & (ndvi > 0.15) & (ndvi <= 0.4)
    labels[agriculture] = 4

    # Urban
    urban = valid & (ndvi < 0.05) & (swir > 0.2)
    labels[urban] = 0

    # NoData
    labels[~valid] = 255

    return labels


def create_labels(year):

    input_file = os.path.join(
        INPUT_DIR,
        f"sentinel2_guwahati_{year}_processed.tif"
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        f"labels_{year}.npy"
    )

    print()
    print("=" * 70)
    print(f"Generating labels for {year}")
    print("=" * 70)

    print("Input:", input_file)
    print("Output:", output_file)

    if not os.path.exists(input_file):
        print()
        print("ERROR: Input file not found!")
        return

    with rasterio.open(input_file) as src:

        print()
        print("Image size:", src.width, "x", src.height)
        print("Number of bands:", src.count)
        print("CRS:", src.crs)

        height = src.height
        width = src.width

        # Create memory-mapped NumPy file.
        # This avoids keeping the complete label image
        # unnecessarily in RAM.
        labels = np.lib.format.open_memmap(
            output_file,
            mode="w+",
            dtype=np.uint8,
            shape=(height, width)
        )

        # Count pixels for each class
        class_counts = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            255: 0
        }

        total_blocks = 0

        # Process image block by block
        for _, window in src.block_windows(1):

            # Band 4 = NIR
            # Band 5 = SWIR1
            # Band 7 = NDVI
            # Band 8 = NDWI

            nir = src.read(4, window=window).astype(np.float32)
            swir = src.read(5, window=window).astype(np.float32)

            ndvi = src.read(7, window=window).astype(np.float32)
            ndwi = src.read(8, window=window).astype(np.float32)

            block_labels = classify_pixels(
                ndvi,
                ndwi,
                nir,
                swir
            )

            row_start = int(window.row_off)
            row_end = row_start + int(window.height)

            col_start = int(window.col_off)
            col_end = col_start + int(window.width)

            labels[
                row_start:row_end,
                col_start:col_end
            ] = block_labels

            # Count classes
            unique, counts = np.unique(
                block_labels,
                return_counts=True
            )

            for value, count in zip(unique, counts):
                class_counts[int(value)] += int(count)

            total_blocks += 1

            if total_blocks % 50 == 0:
                print("Processed blocks:", total_blocks)

        labels.flush()

    # --------------------------------------------------------
    # Print class distribution
    # --------------------------------------------------------

    total_pixels = sum(class_counts.values())

    print()
    print("-" * 70)
    print(f"CLASS DISTRIBUTION - {year}")
    print("-" * 70)

    for class_id, class_name in CLASS_NAMES.items():

        count = class_counts[class_id]

        percentage = (
            (count / total_pixels) * 100
            if total_pixels > 0
            else 0
        )

        print(
            f"{class_id:>3}  "
            f"{class_name:<12} "
            f"{count:>12,} pixels  "
            f"{percentage:>7.2f}%"
        )

    print("-" * 70)
    print("Total pixels:", f"{total_pixels:,}")

    print()
    print("Label file created successfully:")
    print(output_file)


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("GeoSense Phase 2 - Exercise 3.1")
    print("Ground-Truth Labelling")
    print("Study Area: Guwahati")
    print("=" * 70)

    create_labels(2015)
    create_labels(2023)

    print()
    print("=" * 70)
    print("LABEL GENERATION COMPLETE")
    print("=" * 70)