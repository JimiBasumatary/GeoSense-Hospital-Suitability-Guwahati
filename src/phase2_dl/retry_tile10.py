import os
import ee
import geemap
import geopandas as gpd
from shapely.geometry import mapping

# ------------------------------------------------------------
# Initialize Earth Engine
# ------------------------------------------------------------
ee.Initialize(project="avian-branch-478305-f9")

# ------------------------------------------------------------
# Load Guwahati boundary
# ------------------------------------------------------------
boundary_file = r"data\boundaries\guwahati\guwahati.shp"

gdf = gpd.read_file(boundary_file)
gdf = gdf.to_crs("EPSG:4326")

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

print("2023 images available:", collection.size().getInfo())

# ------------------------------------------------------------
# Median composite
# ------------------------------------------------------------
composite = collection.median().clip(roi)

# ------------------------------------------------------------
# Guwahati extent
# ------------------------------------------------------------
min_lon, min_lat, max_lon, max_lat = geometry.bounds

NX = 6
NY = 6

tile_width = (max_lon - min_lon) / NX
tile_height = (max_lat - min_lat) / NY

# Tile 10 = x=3, y=1
tile_number = 10
x = 3
y = 1

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

print()
print("Retrying Tile 10/36")
print(
    "Bounds:",
    tile_min_lon,
    tile_min_lat,
    tile_max_lon,
    tile_max_lat
)

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------
output_file = (
    r"data\satellite\raw\2023_tiles"
    r"\sentinel2_guwahati_2023_tile_10.tif"
)

print("Output:", output_file)
print("Downloading...")

geemap.ee_export_image(
    composite,
    filename=output_file,
    scale=10,
    region=tile,
    file_per_band=False
)

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
    print("TILE 10 DOWNLOAD SUCCESS")
    print("=" * 60)
    print("File:", output_file)
    print("Size: %.2f MB" % size_mb)

else:

    print("ERROR: Tile 10 was not created.")