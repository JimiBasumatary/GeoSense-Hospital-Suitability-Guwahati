from db_connection import get_engine
from sqlalchemy import text


def calculate_commercial_distance():

    print("\n" + "=" * 60)
    print("GeoSense - Calculate Distance to Commercial Land Use")
    print("=" * 60)

    engine = get_engine()

    with engine.begin() as connection:

        # -----------------------------------------------------
        # 1. Create the distance column
        # -----------------------------------------------------
        connection.execute(text("""
            ALTER TABLE candidate_locations
            DROP COLUMN IF EXISTS dist_landuse_commercial_m;
        """))

        connection.execute(text("""
            ALTER TABLE candidate_locations
            ADD COLUMN dist_landuse_commercial_m DOUBLE PRECISION;
        """))

        # -----------------------------------------------------
        # 2. Calculate nearest commercial land-use distance
        # -----------------------------------------------------
        print("\nCalculating nearest commercial distance...")

        connection.execute(text("""
            UPDATE candidate_locations AS c
            SET dist_landuse_commercial_m = (
                SELECT MIN(
                    ST_Distance(
                        c.geometry,
                        l.geometry
                    )
                )
                FROM commercial_landuse_utm AS l
            );
        """))

    # ---------------------------------------------------------
    # 3. Calculate statistics
    # ---------------------------------------------------------

    with engine.connect() as connection:

        total = connection.execute(
            text("""
                SELECT COUNT(*)
                FROM candidate_locations;
            """)
        ).scalar()

        valid = connection.execute(
            text("""
                SELECT COUNT(*)
                FROM candidate_locations
                WHERE dist_landuse_commercial_m IS NOT NULL;
            """)
        ).scalar()

        minimum = connection.execute(
            text("""
                SELECT MIN(dist_landuse_commercial_m)
                FROM candidate_locations;
            """)
        ).scalar()

        maximum = connection.execute(
            text("""
                SELECT MAX(dist_landuse_commercial_m)
                FROM candidate_locations;
            """)
        ).scalar()

        mean = connection.execute(
            text("""
                SELECT AVG(dist_landuse_commercial_m)
                FROM candidate_locations;
            """)
        ).scalar()

    print("\n" + "=" * 60)
    print("Commercial distance calculation completed!")
    print("=" * 60)

    print("Total candidates :", total)
    print("Valid distances  :", valid)
    print("Minimum distance :", round(minimum, 2), "m")
    print("Maximum distance :", round(maximum, 2), "m")
    print("Mean distance    :", round(mean, 2), "m")

    print("\nColumn added:")
    print("dist_landuse_commercial_m")

    print("=" * 60)


if __name__ == "__main__":
    calculate_commercial_distance()