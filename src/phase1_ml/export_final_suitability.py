import os

import geopandas as gpd
from sqlalchemy import create_engine
from dotenv import load_dotenv


OUTPUT = r"data\processed\hospital_suitability_final.gpkg"
LAYER_NAME = "hospital_suitability"


def export_final_layer():

    print("\n" + "=" * 75)
    print("GeoSense - Export Final Hospital Suitability Layer")
    print("=" * 75)

    # ---------------------------------------------------------
    # 1. Load database configuration
    # ---------------------------------------------------------
    print("\n1. Loading database configuration...")

    load_dotenv()

    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")

    connection_string = (
        f"postgresql+psycopg2://"
        f"{DB_USER}:{DB_PASSWORD}@"
        f"{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    engine = create_engine(connection_string)

    # ---------------------------------------------------------
    # 2. Read PostGIS spatial layer
    # ---------------------------------------------------------
    print("\n2. Reading PostGIS suitability layer...")

    sql = """
        SELECT
            location_id,
            longitude,
            latitude,
            suitability_score,
            suitability_class,
            predicted_suitability,
            prediction_confidence,
            geometry
        FROM hospital_suitability_predictions
    """

    gdf = gpd.read_postgis(
        sql,
        engine,
        geom_col="geometry"
    )

    print("Records:", len(gdf))
    print("CRS:", gdf.crs)

    # ---------------------------------------------------------
    # 3. Export GeoPackage
    # ---------------------------------------------------------
    print("\n3. Exporting GeoPackage...")

    if os.path.exists(OUTPUT):
        os.remove(OUTPUT)

    gdf.to_file(
        OUTPUT,
        layer=LAYER_NAME,
        driver="GPKG"
    )

    print("GeoPackage created successfully.")

    # ---------------------------------------------------------
    # 4. Verify exported layer
    # ---------------------------------------------------------
    print("\n4. Verifying exported layer...")

    check = gpd.read_file(
        OUTPUT,
        layer=LAYER_NAME
    )

    print("Exported records:", len(check))
    print("Exported CRS:", check.crs)

    print("\nSuitability classes:")

    print(
        check["predicted_suitability"]
        .value_counts()
        .to_string()
    )

    # ---------------------------------------------------------
    # 5. Final message
    # ---------------------------------------------------------
    print("\n" + "=" * 75)
    print("FINAL GIS LAYER EXPORTED SUCCESSFULLY")
    print("=" * 75)

    print("\nOutput file:")
    print(OUTPUT)

    print("\nLayer:")
    print(LAYER_NAME)

    print("\nRecords:", len(check))

    print("\nUse this layer in ArcGIS Pro to create:")
    print("  • Hospital suitability map")
    print("  • Low suitability locations")
    print("  • Moderate suitability locations")
    print("  • High suitability locations")

    print("\n" + "=" * 75)


if __name__ == "__main__":
    export_final_layer()