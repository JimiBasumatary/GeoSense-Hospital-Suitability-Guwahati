import geopandas as gpd
from sqlalchemy import create_engine

DB_URL = "postgresql+psycopg://postgres:1011@localhost:5432/geosense_db"

POI_FILE = "data/shapefiles/osm/guwahati/guwahati_pois.gpkg"
LANDUSE_FILE = "data/shapefiles/osm/guwahati/guwahati_landuse.gpkg"

engine = create_engine(DB_URL)

print("Connecting to GeoSense database...")

# -------------------------
# Import POIs
# -------------------------
print("\nLoading Guwahati POIs...")
pois = gpd.read_file(POI_FILE)

print("POIs loaded:", len(pois))
print("Original CRS:", pois.crs)

pois = pois.to_crs("EPSG:32646")

pois.to_postgis(
    "osm_pois",
    engine,
    if_exists="replace",
    index=False
)

print("osm_pois created successfully.")

# -------------------------
# Import Land Use
# -------------------------
print("\nLoading Guwahati land use...")
landuse = gpd.read_file(LANDUSE_FILE)

print("Land-use features loaded:", len(landuse))
print("Original CRS:", landuse.crs)

landuse = landuse.to_crs("EPSG:32646")

landuse.to_postgis(
    "osm_landuse",
    engine,
    if_exists="replace",
    index=False
)

print("osm_landuse created successfully.")

print("\n======================================")
print("SUCCESS!")
print("POIs and Land Use imported into PostGIS.")
print("======================================")