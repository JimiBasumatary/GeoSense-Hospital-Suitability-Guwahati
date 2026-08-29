import geopandas as gpd
from sqlalchemy import create_engine
from db_connection import DB_CONFIG
import os

# --------------------------------------------------
# DATABASE CONNECTION
# --------------------------------------------------

engine = create_engine(
    f"postgresql+psycopg2://"
    f"{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}"
    f"/{DB_CONFIG['database']}"
)

print("PostgreSQL connection created successfully.")

# --------------------------------------------------
# OUTPUT PATH
# --------------------------------------------------

output_file = r"data\outputs\geosense_ranked_candidates.gpkg"

os.makedirs(os.path.dirname(output_file), exist_ok=True)

# --------------------------------------------------
# LOAD RANKED CANDIDATES FROM POSTGIS
# --------------------------------------------------

print("\nLoading GeoSense ranked candidates...")

gdf = gpd.read_postgis(
    """
    SELECT
        rank,
        osm_id,
        fclass,
        name,
        type,
        building_area_m2,
        buildings_within_250m,
        dist_road_m,
        area_score,
        density_score,
        road_score,
        suitability_score,
        geometry
    FROM geosense_ranked_candidates
    ORDER BY rank
    """,
    engine,
    geom_col="geometry"
)

print("Candidates loaded:", len(gdf))
print("CRS:", gdf.crs)

# --------------------------------------------------
# SAVE AS GEOPACKAGE
# --------------------------------------------------

print("\nSaving GeoPackage...")

gdf.to_file(
    output_file,
    layer="geosense_ranked_candidates",
    driver="GPKG"
)

print("\n======================================")
print("SUCCESS!")
print("GeoSense ranked candidates exported.")
print("Output:", output_file)
print("Features:", len(gdf))
print("======================================")