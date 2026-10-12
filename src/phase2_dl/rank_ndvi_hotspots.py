
import os
import csv
import numpy as np
import rasterio

IMAGE_2015 = (
    r"data\satellite\processed"
    r"\sentinel2_guwahati_2015_processed.tif"
)
IMAGE_2023 = (
    r"data\satellite\processed"
    r"\sentinel2_guwahati_2023_processed.tif"
)

PATCH_SIZE = 224
SEARCH_STEP = 50
TOP_N = 3
OUTPUT = r"outputs\ndvi_change_hotspots_top3.csv"


def calculate_ndvi(red, nir):
    denominator = nir + red
    return np.divide(
        nir - red,
        denominator,
        out=np.full_like(nir, np.nan, dtype=np.float32),
        where=np.isfinite(denominator) & (denominator != 0)
    )


def patch_mean(array, row, col):
    half = PATCH_SIZE // 2
    patch = array[
        row-half:row+half,
        col-half:col+half
    ]
    valid = np.isfinite(patch)
    if valid.mean() < 0.8:
        return np.nan
    return float(np.nanmean(patch))


print("Ranking NDVI-change hotspots for Guwahati...")

with rasterio.open(IMAGE_2015) as src15, \
     rasterio.open(IMAGE_2023) as src23:

    if (src15.width != src23.width or
        src15.height != src23.height or
        src15.crs != src23.crs or
        src15.transform != src23.transform):
        raise ValueError(
            "The 2015 and 2023 rasters are not aligned. "
            "Align them before comparing NDVI."
        )

    red15 = src15.read(3).astype(np.float32)
    nir15 = src15.read(4).astype(np.float32)
    red23 = src23.read(3).astype(np.float32)
    nir23 = src23.read(4).astype(np.float32)

    ndvi15 = calculate_ndvi(red15, nir15)
    ndvi23 = calculate_ndvi(red23, nir23)

    candidates = []
    half = PATCH_SIZE // 2

    for row in range(half, src23.height-half, SEARCH_STEP):
        for col in range(half, src23.width-half, SEARCH_STEP):
            old = patch_mean(ndvi15, row, col)
            new = patch_mean(ndvi23, row, col)

            if not np.isfinite(old) or not np.isfinite(new):
                continue

            change = new - old
            lon, lat = src23.xy(row, col)

            candidates.append({
                "row": row,
                "col": col,
                "latitude": lat,
                "longitude": lon,
                "ndvi_2015": old,
                "ndvi_2023": new,
                "ndvi_change": change,
                "abs_change": abs(change)
            })

    # Rank by absolute change, but avoid selecting overlapping patches.
    candidates.sort(key=lambda x: x["abs_change"], reverse=True)
    selected = []

    for item in candidates:
        separated = all(
            np.hypot(
                item["row"] - chosen["row"],
                item["col"] - chosen["col"]
            ) >= PATCH_SIZE
            for chosen in selected
        )
        if separated:
            selected.append(item)
        if len(selected) == TOP_N:
            break

if not selected:
    raise RuntimeError("No valid hotspot patches were found.")

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

with open(OUTPUT, "w", newline="", encoding="utf-8-sig") as f:
    fields = [
        "rank", "latitude", "longitude",
        "mean_ndvi_2015", "mean_ndvi_2023",
        "ndvi_change_2023_minus_2015",
        "absolute_ndvi_change", "interpretation",
        "centre_row", "centre_column"
    ]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for rank, item in enumerate(selected, start=1):
        change = item["ndvi_change"]
        if change > 0:
            interpretation = "Potential increase in greenness"
        elif change < 0:
            interpretation = "Potential decrease in greenness"
        else:
            interpretation = "No net NDVI change"

        writer.writerow({
            "rank": rank,
            "latitude": f'{item["latitude"]:.6f}',
            "longitude": f'{item["longitude"]:.6f}',
            "mean_ndvi_2015": f'{item["ndvi_2015"]:.4f}',
            "mean_ndvi_2023": f'{item["ndvi_2023"]:.4f}',
            "ndvi_change_2023_minus_2015": f'{change:.4f}',
            "absolute_ndvi_change": f'{item["abs_change"]:.4f}',
            "interpretation": interpretation,
            "centre_row": item["row"],
            "centre_column": item["col"]
        })

print("\nTOP NDVI-CHANGE HOTSPOTS")
for rank, item in enumerate(selected, start=1):
    print(f"\nRank: {rank}")
    print(f'Location: {item["latitude"]:.6f}, {item["longitude"]:.6f}')
    print(f'NDVI 2015: {item["ndvi_2015"]:.4f}')
    print(f'NDVI 2023: {item["ndvi_2023"]:.4f}')
    print(f'Signed change: {item["ndvi_change"]:.4f}')
    print(f'Absolute change: {item["abs_change"]:.4f}')

print("\nCSV saved to:", OUTPUT)
