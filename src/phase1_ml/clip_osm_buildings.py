import geopandas as gpd
import os

# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

osm_file = r"data\shapefiles\osm\north-eastern-zone-260821-free.gpkg\north-eastern-zone.gpkg"

boundary_file = r"data\boundaries\guwahati\guwahati.shp"

output_file = r"data\shapefiles\osm\guwahati\guwahati_buildings.gpkg"


# --------------------------------------------------
# 1. LOAD GUWAHATI BOUNDARY
# --------------------------------------------------

print("Loading Guwahati boundary...")

boundary = gpd.read_file(boundary_file)

print("Boundary CRS:", boundary.crs)
print("Boundary features:", len(boundary))


# --------------------------------------------------
# 2. CONVERT BOUNDARY TO WGS84
# --------------------------------------------------

boundary = boundary.to_crs("EPSG:4326")

print("Boundary converted to:", boundary.crs)


# --------------------------------------------------
# 3. GET GUWAHATI BOUNDING BOX
# --------------------------------------------------

bbox = tuple(boundary.total_bounds)

print("Guwahati bounding box:")
print(bbox)


# --------------------------------------------------
# 4. LOAD OSM BUILDINGS USING BOUNDING BOX
# --------------------------------------------------

print("\nLoading OSM buildings inside Guwahati bounding box...")
print("This may take some time...")


buildings = gpd.read_file(
    osm_file,
    layer="gis_osm_buildings_a_free",
    bbox=bbox
)


print("Building features loaded:", len(buildings))
print("Building CRS:", buildings.crs)


# --------------------------------------------------
# 5. REPROJECT BOUNDARY TO BUILDING CRS
# --------------------------------------------------

print("\nReprojecting Guwahati boundary to building CRS...")

boundary = boundary.to_crs(buildings.crs)

print("Boundary CRS for clipping:", boundary.crs)
print("Building CRS:", buildings.crs)


# --------------------------------------------------
# 6. CLIP BUILDINGS EXACTLY TO GUWAHATI
# --------------------------------------------------

print("\nClipping buildings to Guwahati boundary...")

clipped = gpd.clip(buildings, boundary)

print("Building features after clipping:", len(clipped))


# --------------------------------------------------
# 7. SAVE CLIPPED BUILDINGS
# --------------------------------------------------

os.makedirs(
    os.path.dirname(output_file),
    exist_ok=True
)

print("\nSaving clipped buildings...")


clipped.to_file(
    output_file,
    layer="guwahati_buildings",
    driver="GPKG"
)


# --------------------------------------------------
# 8. FINAL MESSAGE
# --------------------------------------------------

print("\n======================================")
print("SUCCESS!")
print("Guwahati buildings saved successfully.")
print("Output:", output_file)
print("Features:", len(clipped))
print("======================================")