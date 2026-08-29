import geopandas as gpd
from sqlalchemy import create_engine
from db_connection import DB_CONFIG


# ==========================================================
# GUWAHATI CLIPPED OSM DATA
# ==========================================================

DATA_FOLDER = r"data\shapefiles\osm\guwahati"


LAYERS_TO_LOAD = [
    (
        r"data\shapefiles\osm\guwahati\guwahati_roads.gpkg",
        "guwahati_roads",
        "osm_roads"
    ),
    (
        r"data\shapefiles\osm\guwahati\guwahati_buildings.gpkg",
        "guwahati_buildings",
        "osm_buildings"
    ),
]


def main():

    # ======================================================
    # 1. CREATE POSTGRESQL CONNECTION
    # ======================================================

    engine = create_engine(
        f"postgresql+psycopg2://"
        f"{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}"
        f"/{DB_CONFIG['database']}"
    )

    print("PostgreSQL connection created successfully.")


    # ======================================================
    # 2. LOAD GUWAHATI DATA
    # ======================================================

    for gpkg_path, layer_name, table_name in LAYERS_TO_LOAD:

        print("\n========================================")
        print(f"Loading: {layer_name}")
        print("========================================")

        print(f"File: {gpkg_path}")

        # Read GeoPackage
        gdf = gpd.read_file(
            gpkg_path,
            layer=layer_name
        )

        print(f"Features found: {len(gdf)}")
        print(f"Original CRS: {gdf.crs}")

        # Make sure data uses WGS84
        if gdf.crs is not None:
            gdf = gdf.to_crs(epsg=4326)

        print(f"Final CRS: {gdf.crs}")

        # ==================================================
        # WRITE TO POSTGIS
        # ==================================================

        print(f"Writing to PostGIS table: {table_name}")

        gdf.to_postgis(
            table_name,
            engine,
            if_exists="replace",
            index=False
        )

        print(f"Successfully loaded: {table_name}")
        print(f"Features loaded: {len(gdf)}")


    # ======================================================
    # 3. COMPLETION MESSAGE
    # ======================================================

    print("\n========================================")
    print("GUWAHATI OSM DATA LOADED SUCCESSFULLY!")
    print("========================================")


if __name__ == "__main__":
    main()