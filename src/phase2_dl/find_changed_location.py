import numpy as np
import rasterio


# ============================================================
# GeoSense Phase 2 - Exercise 5.2
# Find a PATCH-LEVEL Changed Location
# Study Area: Guwahati
# ============================================================

IMAGE_2015 = (
    r"data\satellite\processed"
    r"\sentinel2_guwahati_2015_processed.tif"
)

IMAGE_2023 = (
    r"data\satellite\processed"
    r"\sentinel2_guwahati_2023_processed.tif"
)

CHANGE_THRESHOLD = 0.15
PATCH_SIZE = 224

# Search every 50 pixels
SEARCH_STEP = 50

# Keep the complete 224 x 224 patch inside the raster
EDGE_MARGIN = PATCH_SIZE // 2


def patch_mean(array, row, col):

    half = PATCH_SIZE // 2

    row_start = row - half
    row_end = row + half

    col_start = col - half
    col_end = col + half

    patch = array[
        row_start:row_end,
        col_start:col_end
    ]

    return float(
        np.nanmean(patch)
    )


print("=" * 70)
print("GeoSense Phase 2 - Finding PATCH-LEVEL Changed Location")
print("=" * 70)

with rasterio.open(IMAGE_2015) as src15, \
     rasterio.open(IMAGE_2023) as src23:

    print("2015 image:", IMAGE_2015)
    print("2023 image:", IMAGE_2023)

    print(
        "Image size:",
        src15.width,
        "x",
        src15.height
    )

    # --------------------------------------------------------
    # Read Red and NIR
    # --------------------------------------------------------

    red15 = src15.read(3).astype(np.float32)
    nir15 = src15.read(4).astype(np.float32)

    red23 = src23.read(3).astype(np.float32)
    nir23 = src23.read(4).astype(np.float32)

    # --------------------------------------------------------
    # Calculate pixel-level NDVI
    # --------------------------------------------------------

    den15 = nir15 + red15

    ndvi15 = np.divide(
        nir15 - red15,
        den15,
        out=np.full_like(nir15, np.nan),
        where=den15 != 0
    )

    den23 = nir23 + red23

    ndvi23 = np.divide(
        nir23 - red23,
        den23,
        out=np.full_like(nir23, np.nan),
        where=den23 != 0
    )

    print()
    print("Searching 224 x 224 patches...")
    print("Search step:", SEARCH_STEP, "pixels")
    print("Change threshold:", CHANGE_THRESHOLD)

    best_difference = -1
    best_row = None
    best_col = None
    best_ndvi15 = None
    best_ndvi23 = None

    # --------------------------------------------------------
    # Search interior patch centres
    # --------------------------------------------------------

    for row in range(
        EDGE_MARGIN,
        src23.height - EDGE_MARGIN,
        SEARCH_STEP
    ):

        for col in range(
            EDGE_MARGIN,
            src23.width - EDGE_MARGIN,
            SEARCH_STEP
        ):

            ndvi_2015 = patch_mean(
                ndvi15,
                row,
                col
            )

            ndvi_2023 = patch_mean(
                ndvi23,
                row,
                col
            )

            if not np.isfinite(ndvi_2015):
                continue

            if not np.isfinite(ndvi_2023):
                continue

            difference = abs(
                ndvi_2023 - ndvi_2015
            )

            if difference > best_difference:

                best_difference = difference
                best_row = row
                best_col = col
                best_ndvi15 = ndvi_2015
                best_ndvi23 = ndvi_2023

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    if best_row is None:

        print()
        print("No valid patch found.")

    else:

        lon, lat = src23.xy(
            best_row,
            best_col
        )

        print()
        print("=" * 70)
        print("PATCH-LEVEL CHANGED LOCATION FOUND")
        print("=" * 70)

        print("Latitude:", lat)
        print("Longitude:", lon)

        print("Patch centre row:", best_row)
        print("Patch centre column:", best_col)

        print()
        print(
            "224 x 224 mean NDVI 2015:",
            round(best_ndvi15, 4)
        )

        print(
            "224 x 224 mean NDVI 2023:",
            round(best_ndvi23, 4)
        )

        print(
            "Absolute NDVI difference:",
            round(best_difference, 4)
        )

        print(
            "Change threshold:",
            CHANGE_THRESHOLD
        )

        print(
            "Change flag:",
            best_difference > CHANGE_THRESHOLD
        )

        print()
        print("=" * 70)
        print(
            "This coordinate should now be tested"
        )
        print(
            "directly with image_classifier.py."
        )
        print("=" * 70)