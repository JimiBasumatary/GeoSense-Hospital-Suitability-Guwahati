import geopandas as gpd
from sqlalchemy import create_engine
from db_connection import DB_CONFIG


# ============================================================
# CONFIGURATION
# ============================================================

LOCATION_NAME = "Guwahati"
OUTPUT_TABLE = "building_candidates"

# Guwahati is in UTM Zone 46N
PROJECTED_CRS = "EPSG:32646"
WGS84_CRS = "EPSG:4326"

# Candidate search radius
SEARCH_RADIUS_METERS = 1000


# ============================================================
# 1. CREATE POSTGRESQL CONNECTION
# ============================================================

engine = create_engine(
    f"postgresql+psycopg2://"
    f"{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}"
    f"/{DB_CONFIG['database']}"
)

print("PostgreSQL connection created successfully.")


# ============================================================
# 2. LOAD OSM BUILDINGS
# ============================================================

print("\nLoading OSM buildings...")

buildings = gpd.read_postgis(
    """
    SELECT
        osm_id,
        code,
        fclass,
        name,
        type,
        geometry
    FROM osm_buildings
    """,
    engine,
    geom_col="geometry"
)

print("Buildings loaded:", len(buildings))
print("Buildings CRS:", buildings.crs)


# ============================================================
# 3. LOAD GUWAHATI LOCATION
# ============================================================

print("\nLoading Guwahati location...")

location = gpd.read_postgis(
    """
    SELECT
        id,
        name,
        geom
    FROM locations
    WHERE name = 'Guwahati'
    """,
    engine,
    geom_col="geom"
)

print("Location loaded:", len(location))
print("Location CRS:", location.crs)


# ============================================================
# 4. CHECK LOCATION
# ============================================================

if len(location) == 0:
    raise ValueError("Guwahati location was not found in locations table.")


# ============================================================
# 5. REPROJECT BOTH DATASETS TO PROJECTED CRS
# ============================================================

print("\nReprojecting data to UTM Zone 46N...")

buildings_projected = buildings.to_crs(PROJECTED_CRS)

location_projected = location.to_crs(PROJECTED_CRS)

print("Projected buildings CRS:", buildings_projected.crs)
print("Projected location CRS:", location_projected.crs)


# ============================================================
# 6. CREATE 1 KM SEARCH AREA
# ============================================================

print("\nCreating 1 km candidate search area...")

search_area = location_projected.copy()

search_area["geometry"] = (
    search_area["geom"].buffer(SEARCH_RADIUS_METERS)
)

# Make sure GeoPandas uses geometry as the active geometry
search_area = search_area.set_geometry("geometry")

print("Search radius:", SEARCH_RADIUS_METERS, "meters")


# ============================================================
# 7. FIND BUILDINGS INTERSECTING SEARCH AREA
# ============================================================

print("\nFinding candidate buildings...")

candidates = gpd.sjoin(
    buildings_projected,
    search_area[["geometry"]],
    predicate="intersects",
    how="inner"
)

print("Candidate buildings found:", len(candidates))


# ============================================================
# 8. REMOVE DUPLICATE BUILDINGS
# ============================================================

candidates = candidates.drop_duplicates(
    subset=["osm_id"]
)

print("Unique candidate buildings:", len(candidates))


# ============================================================
# 9. RETURN CANDIDATES TO WGS84
# ============================================================

print("\nConverting candidates back to WGS84...")

candidates = candidates.to_crs(WGS84_CRS)


# ============================================================
# 10. KEEP REQUIRED COLUMNS
# ============================================================

candidates = candidates[
    [
        "osm_id",
        "code",
        "fclass",
        "name",
        "type",
        "geometry"
    ]
]


# ============================================================
# 11. SAVE TO POSTGIS
# ============================================================

print("\nSaving candidates to PostGIS...")

candidates.to_postgis(
    OUTPUT_TABLE,
    engine,
    if_exists="replace",
    index=False
)

print("Successfully created table:", OUTPUT_TABLE)


# ============================================================
# 12. CREATE SPATIAL INDEX
# ============================================================

print("\nCreating spatial index...")

with engine.begin() as connection:

    connection.exec_driver_sql(
        f"""
        CREATE INDEX IF NOT EXISTS
        idx_{OUTPUT_TABLE}_geometry
        ON {OUTPUT_TABLE}
        USING GIST (geometry);
        """
    )


# ============================================================
# 13. FINAL RESULT
# ============================================================

print("\n==============================================")
print("CANDIDATE GENERATION SUCCESSFUL!")
print("==============================================")
print("Location:", LOCATION_NAME)
print("Search radius:", SEARCH_RADIUS_METERS, "meters")
print("Candidate buildings:", len(candidates))
print("PostGIS table:", OUTPUT_TABLE)
print("CRS:", WGS84_CRS)
print("==============================================")