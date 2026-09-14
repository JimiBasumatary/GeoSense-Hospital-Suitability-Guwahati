from db_connection import get_engine
from sqlalchemy import text


def assign_landuse_class():

    print("\n" + "=" * 60)
    print("GeoSense - Assign Land Use Class")
    print("Direct intersection + nearest within 500 m")
    print("=" * 60)

    engine = get_engine()

    with engine.begin() as connection:

        # -----------------------------------------------------
        # 1. Reset existing classification
        # -----------------------------------------------------

        connection.execute(text("""
            UPDATE candidate_locations
            SET land_use_class = NULL;
        """))

        # -----------------------------------------------------
        # 2. Direct intersection
        # -----------------------------------------------------

        print("\nStep 1: Assigning direct land-use intersections...")

        connection.execute(text("""
            UPDATE candidate_locations AS c
            SET land_use_class = l.fclass
            FROM guwahati_landuse_utm AS l
            WHERE ST_Intersects(c.geometry, l.geometry);
        """))

        # -----------------------------------------------------
        # 3. Find nearest land-use polygon within 500 m
        # -----------------------------------------------------

        print("Step 2: Finding nearest land use within 500 m...")

        connection.execute(text("""
            DROP TABLE IF EXISTS nearest_landuse_500m;
        """))

        connection.execute(text("""
            CREATE TEMP TABLE nearest_landuse_500m AS

            SELECT DISTINCT ON (c.location_id)
                c.location_id,
                l.fclass
            FROM candidate_locations AS c
            CROSS JOIN guwahati_landuse_utm AS l
            WHERE c.land_use_class IS NULL
              AND ST_DWithin(
                    c.geometry,
                    l.geometry,
                    500
              )
            ORDER BY
                c.location_id,
                c.geometry <-> l.geometry;
        """))

        # -----------------------------------------------------
        # 4. Update unclassified candidates
        # -----------------------------------------------------

        print("Step 3: Updating unclassified candidates...")

        connection.execute(text("""
            UPDATE candidate_locations AS c
            SET land_use_class = n.fclass
            FROM nearest_landuse_500m AS n
            WHERE c.location_id = n.location_id;
        """))

        # -----------------------------------------------------
        # 5. Remove temporary table
        # -----------------------------------------------------

        connection.execute(text("""
            DROP TABLE IF EXISTS nearest_landuse_500m;
        """))

    # ---------------------------------------------------------
    # 6. Check results
    # ---------------------------------------------------------

    with engine.connect() as connection:

        total = connection.execute(
            text("""
                SELECT COUNT(*)
                FROM candidate_locations;
            """)
        ).scalar()

        classified = connection.execute(
            text("""
                SELECT COUNT(*)
                FROM candidate_locations
                WHERE land_use_class IS NOT NULL;
            """)
        ).scalar()

        unclassified = connection.execute(
            text("""
                SELECT COUNT(*)
                FROM candidate_locations
                WHERE land_use_class IS NULL;
            """)
        ).scalar()

        rows = connection.execute(
            text("""
                SELECT
                    land_use_class,
                    COUNT(*) AS candidate_count
                FROM candidate_locations
                GROUP BY land_use_class
                ORDER BY candidate_count DESC;
            """)
        ).fetchall()

    print("\n" + "=" * 60)
    print("Land-use classification completed!")
    print("=" * 60)

    print("Total candidates :", total)
    print("Classified       :", classified)
    print("Unclassified     :", unclassified)

    print("\nLand-use distribution:")

    for row in rows:
        print(f"{str(row.land_use_class):25} {row.candidate_count}")

    print("=" * 60)


if __name__ == "__main__":
    assign_landuse_class()