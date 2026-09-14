from db_connection import get_engine
from sqlalchemy import text


def verify_features():

    print("\n" + "=" * 70)
    print("GeoSense - Verify Exercise 4 Feature Table")
    print("=" * 70)

    engine = get_engine()

    with engine.connect() as connection:

        # -----------------------------------------------------
        # 1. Check table columns
        # -----------------------------------------------------

        print("\nFeature columns:")

        columns = connection.execute(
            text("""
                SELECT
                    column_name,
                    data_type
                FROM information_schema.columns
                WHERE table_name = 'candidate_locations'
                ORDER BY ordinal_position;
            """)
        ).fetchall()

        for column in columns:
            print(f"{column.column_name:35} {column.data_type}")

        # -----------------------------------------------------
        # 2. Count candidate locations
        # -----------------------------------------------------

        total = connection.execute(
            text("""
                SELECT COUNT(*)
                FROM candidate_locations;
            """)
        ).scalar()

        print("\nTotal candidate locations:", total)

        # -----------------------------------------------------
        # 3. Check NULL values
        # -----------------------------------------------------

        print("\nNULL value check:")

        feature_columns = [
            "dist_road_m",
            "dist_hospital_m",
            "flood_risk",
            "population_value",
            "land_use_class",
            "elevation_m",
            "dist_landuse_commercial_m"
        ]

        for column in feature_columns:

            query = text(f"""
                SELECT COUNT(*)
                FROM candidate_locations
                WHERE {column} IS NULL;
            """)

            null_count = connection.execute(query).scalar()

            print(f"{column:35} {null_count}")

        # -----------------------------------------------------
        # 4. Show sample records
        # -----------------------------------------------------

        print("\nSample candidate records:")

        rows = connection.execute(
            text("""
                SELECT
                    location_id,
                    ROUND(CAST(ST_Y(ST_Transform(geometry, 4326)) AS numeric), 6)
                        AS latitude,
                    ROUND(CAST(ST_X(ST_Transform(geometry, 4326)) AS numeric), 6)
                        AS longitude,
                    ROUND(CAST(dist_road_m AS numeric), 2)
                        AS dist_road_m,
                    ROUND(CAST(dist_hospital_m AS numeric), 2)
                        AS dist_hospital_m,
                    flood_risk,
                    population_value,
                    land_use_class,
                    elevation_m,
                    ROUND(CAST(dist_landuse_commercial_m AS numeric), 2)
                        AS dist_landuse_commercial_m
                FROM candidate_locations
                ORDER BY location_id
                LIMIT 10;
            """)
        ).fetchall()

        for row in rows:
            print(row)

    print("\n" + "=" * 70)
    print("Verification completed.")
    print("=" * 70)


if __name__ == "__main__":
    verify_features()