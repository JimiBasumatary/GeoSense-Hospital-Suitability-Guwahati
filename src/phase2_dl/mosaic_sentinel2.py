import os
import glob
import rasterio
from rasterio.merge import merge

# ============================================================
# GeoSense Agent 2.0 - Phase 2
# Sentinel-2 2023 Mosaic
# Study Area: Guwahati
# ============================================================

input_dir = r"data\satellite\raw\2023_tiles"

output_dir = r"data\satellite\raw"

output_file = os.path.join(
    output_dir,
    "sentinel2_guwahati_2023_mosaic.tif"
)

# ------------------------------------------------------------
# Find all tiles
# ------------------------------------------------------------
tile_files = sorted(
    glob.glob(
        os.path.join(
            input_dir,
            "sentinel2_guwahati_2023_tile_*.tif"
        )
    )
)

print("=" * 60)
print("GeoSense Phase 2 - Sentinel-2 Mosaic")
print("=" * 60)

print("Tiles found:", len(tile_files))

if len(tile_files) != 36:
    raise RuntimeError(
        f"Expected 36 tiles, found {len(tile_files)}"
    )

# ------------------------------------------------------------
# Open tiles
# ------------------------------------------------------------
src_files = []

for file in tile_files:

    src = rasterio.open(file)
    src_files.append(src)

print("All 36 tiles opened successfully.")

# ------------------------------------------------------------
# Create mosaic
# ------------------------------------------------------------
print()
print("Creating mosaic...")

mosaic, transform = merge(src_files)

# ------------------------------------------------------------
# Update metadata
# ------------------------------------------------------------
out_meta = src_files[0].meta.copy()

out_meta.update({
    "driver": "GTiff",
    "height": mosaic.shape[1],
    "width": mosaic.shape[2],
    "transform": transform,
    "count": mosaic.shape[0]
})

# ------------------------------------------------------------
# Write mosaic
# ------------------------------------------------------------
print("Writing:", output_file)

with rasterio.open(
    output_file,
    "w",
    **out_meta
) as dest:

    dest.write(mosaic)

# ------------------------------------------------------------
# Close files
# ------------------------------------------------------------
for src in src_files:
    src.close()

# ------------------------------------------------------------
# Verify
# ------------------------------------------------------------
if os.path.exists(output_file):

    size_mb = (
        os.path.getsize(output_file)
        / (1024 * 1024)
    )

    print()
    print("=" * 60)
    print("MOSAIC CREATED SUCCESSFULLY")
    print("=" * 60)
    print("File:", output_file)
    print("Size: %.2f MB" % size_mb)

else:

    print("ERROR: Mosaic was not created.")