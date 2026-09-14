import pandas as pd
from db_connection import get_engine
from sqlalchemy import text


OUTPUT = r"data\processed\features.csv"


def export_features():

    print("\n" + "=" * 70)
    print("GeoSense - Export Exercise 4 Feature Dataset")
    print("=" * 70)

    engine = get_engine()

    # ---------------------------------------------------------
    # 1. Read candidate features from PostGIS
    # ---------------------------------------------------------

    print("\nReading feature table from PostGIS...")

    sql = """
        SELECT
            location_id,
            longitude,
            latitude,
            dist_road_m,
            dist_hospital_m,
            flood_risk,
            population_value,
            land_use_class,
            elevation_m,
            dist_landuse_commercial_m
        FROM candidate_locations
        ORDER BY location_id;
    """

    df = pd.read_sql(text(sql), engine)

    # ---------------------------------------------------------
    # 2. Save CSV
    # ---------------------------------------------------------

    df.to_csv(OUTPUT, index=False)

    # ---------------------------------------------------------
    # 3. Basic information
    # ---------------------------------------------------------

    print("\nExport completed successfully!")

    print("Rows :", len(df))
    print("Columns :", len(df.columns))

    print("\nColumns:")
    for column in df.columns:
        print(" -", column)

    print("\nMissing values:")

    print(df.isna().sum())

    print("\nOutput file:")
    print(OUTPUT)

    print("=" * 70)


if __name__ == "__main__":
    export_features()