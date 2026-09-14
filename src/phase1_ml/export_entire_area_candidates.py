import geopandas as gpd
import os


INPUT_FILE = "data/outputs/guwahati_emergency_shelter_predictions.gpkg"

OUTPUT_FILE = (
    "data/outputs/"
    "guwahati_final_ranked_emergency_shelter_candidates.gpkg"
)


print("Loading entire-area emergency shelter predictions...")

gdf = gpd.read_file(INPUT_FILE)

print("Total buildings loaded:", len(gdf))


# --------------------------------------------------
# SELECT SUITABLE CANDIDATES
# --------------------------------------------------

suitable = gdf[
    gdf["emergency_shelter_prediction"] == 1
].copy()

print("Suitable candidates:", len(suitable))


# --------------------------------------------------
# SORT BY SUITABILITY RANK
# --------------------------------------------------

suitable = suitable.sort_values(
    "emergency_shelter_rank"
)


# --------------------------------------------------
# SELECT FINAL IMPORTANT COLUMNS
# --------------------------------------------------

final_columns = [
    "emergency_shelter_rank",
    "osm_id",
    "fclass",
    "name",
    "type",
    "building_area_m2",
    "buildings_within_250m",
    "dist_road_m",
    "pois_within_500m",
    "dominant_landuse",
    "dominant_landuse_area_m2",
    "emergency_shelter_probability",
    "geometry"
]

final_gdf = suitable[final_columns].copy()


# --------------------------------------------------
# CREATE OUTPUT DIRECTORY
# --------------------------------------------------

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


# --------------------------------------------------
# EXPORT GEOPACKAGE
# --------------------------------------------------

print("\nSaving final ranked candidates...")

final_gdf.to_file(
    OUTPUT_FILE,
    layer="final_ranked_candidates",
    driver="GPKG"
)


# --------------------------------------------------
# VERIFICATION
# --------------------------------------------------

print("\n========================================")
print("SUCCESS!")
print("Entire-area ranked candidates exported.")
print("========================================")

print("Total buildings analysed:", len(gdf))
print("Suitable candidates exported:", len(final_gdf))

print(
    "Rank range:",
    int(final_gdf["emergency_shelter_rank"].min()),
    "to",
    int(final_gdf["emergency_shelter_rank"].max())
)

print("CRS:", final_gdf.crs)

print("Output:", OUTPUT_FILE)

print("========================================")