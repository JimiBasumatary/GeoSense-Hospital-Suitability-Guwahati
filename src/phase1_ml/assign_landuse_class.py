from db_connection import get_engine
from sqlalchemy import text


def assign_landuse_class():

    print("\n" + "=" * 60)
    print("GeoSense - Assign Land Use Class")
    print("=" * 60)

    engine = get_engine()

    with engine.begin() as connection:

        # -----------------------------------------------------
        # 1. Add land-use class column
        # -----------------------------------------------------
        connection.execute(text("""
            ALTER TABLE candidate_locations
            DROP COLUMN IF EXISTS land_use_class;
        """))

        connection.execute(text("""
            ALTER TABLE candidate_locations
            ADD COLUMN land_use_class VARCHAR(100);
        """))

        # -----------------------------------------------------
        # 2. Assign land-use class using spatial intersection
        # -----------------------------------------------------
        print("\nAssigning land-use classes...")

        connection.execute(text("""
            UPDATE candidate_locations AS c
            SET land_use_class = l.fclass
            FROM guwahati_landuse_utm AS l
            WHERE ST_Intersects(
                c.geometry,
                l.geometry
            );
        """))

    # ---------------------------------------------------------
    # 3. Check results
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