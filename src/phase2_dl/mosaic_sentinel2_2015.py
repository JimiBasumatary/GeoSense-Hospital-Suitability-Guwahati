import os
import glob
import rasterio
from rasterio.merge import merge

# ============================================================
# GeoSense Agent 2.0 - Phase 2
# Sentinel-2 2015 Mosaic
# Study Area: Guwahati
# ============================================================

input_dir = r"data\satellite\raw\2015_tiles"

output_dir = r"data\satellite\raw"

output_file = os.path.join(
    output_dir,
    "sentinel2_guwahati_2015_mosaic.tif"
)

print("=" * 70)
print("GeoSense Phase 2 - Sentinel-2 2015 Mosaic")
print("=" * 70)

# ------------------------------------------------------------
# Find all 2015 tiles
# ------------------------------------------------------------

tile_pattern = os.path.join(
    input_dir,
    "sentinel2_guwahati_2015_tile_*.tif"
)

tile_files = sorted(
    glob.glob(tile_pattern)
)

print()
print("Input folder:", input_dir)
print("Tiles found:", len(tile_files))

if len(tile_files) != 36:

    raise RuntimeError(
        f"Expected 36 tiles, but found {len(tile_files)}."
    )

# ------------------------------------------------------------
# Open all tiles
# ------------------------------------------------------------

src_files = []

try:

    for tile_file in tile_files:

        print(
            "Opening:",
            os.path.basename(tile_file)
        )

        src = rasterio.open(tile_file)

        src_files.append(src)

    # --------------------------------------------------------
    # Create mosaic
    # --------------------------------------------------------

    print()
    print("Creating 2015 mosaic...")

    mosaic, out_transform = merge(
        src_files
    )

    print(
        "Mosaic shape:",
        mosaic.shape
    )

    # --------------------------------------------------------
    # Update raster profile
    # --------------------------------------------------------

    out_meta = src_files[0].meta.copy()

    out_meta.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform,
            "count": mosaic.shape[0],
            "compress": "deflate"
        }
    )

    # --------------------------------------------------------
    # Write mosaic
    # --------------------------------------------------------

    print()
    print(
        "Writing:",
        output_file
    )

    with rasterio.open(
        output_file,
        "w",
        **out_meta
    ) as dest:

        dest.write(mosaic)

finally:

    for src in src_files:

        src.close()

# ------------------------------------------------------------
# Final validation
# ------------------------------------------------------------

print()
print("=" * 70)
print("2015 MOSAIC COMPLETE")
print("=" * 70)

print(
    "Output:",
    output_file
)

with rasterio.open(output_file) as src:

    print(
        "CRS:",
        src.crs
    )

    print(
        "Size:",
        src.width,
        "x",
        src.height
    )

    print(
        "Bands:",
        src.count
    )

    print(
        "Resolution:",
        src.res
    )

    print(
        "Bounds:",
        src.bounds
    )

print()
print(
    "2015 Sentinel-2 mosaic created successfully."
)