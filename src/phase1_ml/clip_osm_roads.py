import geopandas as gpd
import os

# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

osm_file = r"data\shapefiles\osm\north-eastern-zone-260821-free.gpkg\north-eastern-zone.gpkg"
boundary_file = r"data\boundaries\guwahati\guwahati.shp"
output_file = r"data\shapefiles\osm\guwahati\guwahati_roads.gpkg"

# --------------------------------------------------
# 1. LOAD GUWAHATI BOUNDARY
# --------------------------------------------------

print("Loading Guwahati boundary...")

boundary = gpd.read_file(boundary_file)

print("Boundary CRS:", boundary.crs)
print("Boundary features:", len(boundary))

# --------------------------------------------------
# 2. CONVERT BOUNDARY TO WGS84 FOR OSM BOUNDING BOX
# --------------------------------------------------

boundary_wgs84 = boundary.to_crs("EPSG:4326")

print("Boundary converted to WGS84:", boundary_wgs84.crs)

# --------------------------------------------------
# 3. GET BOUNDING BOX
# --------------------------------------------------

bbox = tuple(boundary_wgs84.total_bounds)

print("Guwahati bounding box:")
print(bbox)

# --------------------------------------------------
# 4. LOAD ONLY ROADS INSIDE BOUNDING BOX
# --------------------------------------------------

print("\nLoading OSM roads inside Guwahati bounding box...")
print("This may take some time...")

roads = gpd.read_file(
    osm_file,
    layer="gis_osm_roads_free",
    bbox=bbox
)

print("Road features loaded:", len(roads))
print("OSM roads CRS:", roads.crs)

# --------------------------------------------------
# 5. REPROJECT BOUNDARY TO EXACT OSM ROAD CRS
# --------------------------------------------------

print("\nReprojecting Guwahati boundary to OSM road CRS...")

boundary_for_clip = boundary.to_crs(roads.crs)

print("Boundary CRS for clipping:", boundary_for_clip.crs)
print("Road CRS:", roads.crs)

# --------------------------------------------------
# 6. CLIP EXACTLY TO GUWAHATI BOUNDARY
# --------------------------------------------------

print("\nClipping roads to Guwahati boundary...")

clipped = gpd.clip(roads, boundary_for_clip)

print("Road features after clipping:", len(clipped))

# --------------------------------------------------
# 7. SAVE RESULT
# --------------------------------------------------

os.makedirs(os.path.dirname(output_file), exist_ok=True)

print("\nSaving clipped roads...")

clipped.to_file(
    output_file,
    layer="guwahati_roads",
    driver="GPKG"
)

# --------------------------------------------------
# 8. FINAL MESSAGE
# --------------------------------------------------

print("\n======================================")
print("SUCCESS!")
print("Guwahati roads saved successfully.")
print("Output:", output_file)
print("Features:", len(clipped))
print("======================================")