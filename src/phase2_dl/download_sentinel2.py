import os
import ee
import geemap
import geopandas as gpd
from shapely.geometry import mapping
from dotenv import load_dotenv

# ============================================================
# GeoSense Agent 2.0 - Phase 2
# Exercise 2: Sentinel-2 Tiled Acquisition
# Study Area: Guwahati
# ============================================================

load_dotenv()

print("=" * 70)
print("GeoSense Phase 2 - Sentinel-2 Tiled Acquisition")
print("=" * 70)

# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------
boundary_file = r"data\boundaries\guwahati\guwahati.shp"

output_base = r"data\satellite\raw"
tile_dir = os.path.join(output_base, "2023_tiles")

os.makedirs(tile_dir, exist_ok=True)

# ------------------------------------------------------------
# Load Guwahati boundary
# ------------------------------------------------------------
print()
print("Loading Guwahati boundary...")

gdf = gpd.read_file(boundary_file)

print("Boundary CRS:", gdf.crs)

# Convert to WGS84 for Earth Engine
gdf = gdf.to_crs("EPSG:4326")

print("Boundary converted to:", gdf.crs)

# Combine all boundary geometries
geometry = gdf.geometry.union_all()

# ------------------------------------------------------------
# Earth Engine
# ------------------------------------------------------------
ee.Initialize(project="avian-branch-478305-f9")

print("Earth Engine: INITIALIZED")

# Create Earth Engine ROI AFTER initialization
roi = ee.Geometry(mapping(geometry))

# ------------------------------------------------------------
# Load Guwahati boundary
# ------------------------------------------------------------
print()
print("Loading Guwahati boundary...")

gdf = gpd.read_file(boundary_file)

print("Boundary CRS:", gdf.crs)

# Convert to WGS84 for Earth Engine
gdf = gdf.to_crs("EPSG:4326")

print("Boundary converted to:", gdf.crs)

# Combine all boundary geometries
geometry = gdf.geometry.union_all()

roi = ee.Geometry(mapping(geometry))

# ------------------------------------------------------------
# Sentinel-2 bands
# ------------------------------------------------------------
BANDS = [
    "B2",
    "B3",
    "B4",
    "B8",
    "B11",
    "B12"
]

# ------------------------------------------------------------
# 2023 Sentinel-2 collection
# ------------------------------------------------------------
print()
print("Searching Sentinel-2 imagery for 2023...")

collection = (
    ee.ImageCollection(
        "COPERNICUS/S2_SR_HARMONIZED"
    )
    .filterBounds(roi)
    .filterDate(
        "2023-01-01",
        "2023-12-31"
    )
    .filter(
        ee.Filter.lt(
            "CLOUDY_PIXEL_PERCENTAGE",
            10
        )
    )
    .select(BANDS)
)

count = collection.size().getInfo()

print("2023 images available:", count)

if count == 0:
    raise RuntimeError(
        "No Sentinel-2 images found."
    )

# ------------------------------------------------------------
# Median composite
# ------------------------------------------------------------
print()
print("Creating median composite...")

composite = collection.median()

# Clip to actual Guwahati boundary
composite = composite.clip(roi)

# ------------------------------------------------------------
# Get boundary extent
# ------------------------------------------------------------
bounds = geometry.bounds

min_lon, min_lat, max_lon, max_lat = bounds

print()
print("Guwahati boundary extent:")
print(
    min_lon,
    min_lat,
    max_lon,
    max_lat
)

# ------------------------------------------------------------
# 6 x 6 tiled download
# ------------------------------------------------------------
NX = 6
NY = 6

tile_width = (max_lon - min_lon) / NX
tile_height = (max_lat - min_lat) / NY

total_tiles = NX * NY

print()
print("Tile grid:", NX, "x", NY)
print("Total tiles:", total_tiles)
print()

successful = 0
failed = 0

# ------------------------------------------------------------
# Download tiles
# ------------------------------------------------------------
for y in range(NY):

    for x in range(NX):

        tile_number = y * NX + x + 1

        tile_min_lon = min_lon + x * tile_width
        tile_max_lon = min_lon + (x + 1) * tile_width

        tile_min_lat = min_lat + y * tile_height
        tile_max_lat = min_lat + (y + 1) * tile_height

        tile = ee.Geometry.Rectangle([
            tile_min_lon,
            tile_min_lat,
            tile_max_lon,
            tile_max_lat
        ])

        output_file = os.path.join(
            tile_dir,
            f"sentinel2_guwahati_2023_tile_{tile_number:02d}.tif"
        )

        print("-" * 70)
        print(
            f"Tile {tile_number}/{total_tiles}"
        )

        print(
            "Bounds:",
            tile_min_lon,
            tile_min_lat,
            tile_max_lon,
            tile_max_lat
        )

        # Skip if already downloaded
        if os.path.exists(output_file):

            print("Already exists - skipping.")

            successful += 1

            continue

        try:

            geemap.ee_export_image(
                composite,
                filename=output_file,
                scale=10,
                region=tile,
                file_per_band=False
            )

            if os.path.exists(output_file):

                size_mb = (
                    os.path.getsize(output_file)
                    / (1024 * 1024)
                )

                print(
                    f"SUCCESS - {size_mb:.2f} MB"
                )

                successful += 1

            else:

                print("FAILED - file not created.")

                failed += 1

        except Exception as e:

            print("FAILED")
            print("Error:", e)

            failed += 1

# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------
print()
print("=" * 70)
print("2023 SENTINEL-2 DOWNLOAD SUMMARY")
print("=" * 70)

print("Study Area: Guwahati")
print("Images used:", count)
print("Bands:", BANDS)
print("Resolution: 10 metres")
print("Cloud threshold: <10%")
print("Composite: Median")
print("Total tiles:", total_tiles)
print("Successful:", successful)
print("Failed:", failed)
print("Output folder:", tile_dir)

if failed == 0:

    print()
    print("ALL 2023 TILES DOWNLOADED SUCCESSFULLY")

else:

    print()
    print(
        "Some tiles failed. Failed tiles can be retried."
    )