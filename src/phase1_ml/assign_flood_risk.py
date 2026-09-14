from db_connection import get_engine
from sqlalchemy import text


def assign_flood_risk():

    print("\n" + "=" * 60)
    print("GeoSense - Assign Flood Risk")
    print("=" * 60)

    engine = get_engine()

    with engine.begin() as connection:

        # -----------------------------------------------------
        # 1. Check whether flood_zones exists
        # -----------------------------------------------------

        table_exists = connection.execute(
            text("""
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'flood_zones'
                );
            """)
        ).scalar()

        # -----------------------------------------------------
        # 2. If flood_zones exists, use it
        # -----------------------------------------------------

        if table_exists:

            print("\nflood_zones table found.")

            connection.execute(text("""
                ALTER TABLE candidate_locations
                DROP COLUMN IF EXISTS flood_risk;
            """))

            connection.execute(text("""
                ALTER TABLE candidate_locations
                ADD COLUMN flood_risk DOUBLE PRECISION;
            """))

            connection.execute(text("""
                UPDATE candidate_locations AS c
                SET flood_risk =
                    CASE
                        WHEN LOWER(f.risk_level) = 'high' THEN 8
                        WHEN LOWER(f.risk_level) = 'medium' THEN 5
                        WHEN LOWER(f.risk_level) = 'low' THEN 2
                        ELSE 0
                    END
                FROM flood_zones AS f
                WHERE ST_Within(c.geometry, f.geometry);
            """))

            connection.execute(text("""
                UPDATE candidate_locations
                SET flood_risk = 0
                WHERE flood_risk IS NULL;
            """))

            print("Flood risk calculated from flood_zones.")

        # -----------------------------------------------------
        # 3. If flood_zones does NOT exist, use placeholder
        # -----------------------------------------------------

        else:

            print("\nflood_zones table not found.")
            print("Using 0 as an explicit placeholder.")
            print("This does NOT mean zero actual flood risk.")

            connection.execute(text("""
                ALTER TABLE candidate_locations
                DROP COLUMN IF EXISTS flood_risk;
            """))

            connection.execute(text("""
                ALTER TABLE candidate_locations
                ADD COLUMN flood_risk DOUBLE PRECISION;
            """))

            connection.execute(text("""
                UPDATE candidate_locations
                SET flood_risk = 0;
            """))

    # ---------------------------------------------------------
    # 4. Verify
    # ---------------------------------------------------------

    with engine.connect() as connection:

        total = connection.execute(
            text("""
                SELECT COUNT(*)
                FROM candidate_locations;
            """)
        ).scalar()

        zero_count = connection.execute(
            text("""
                SELECT COUNT(*)
                FROM candidate_locations
                WHERE flood_risk = 0;
            """)
        ).scalar()

        minimum = connection.execute(
            text("""
                SELECT MIN(flood_risk)
                FROM candidate_locations;
            """)
        ).scalar()

        maximum = connection.execute(
            text("""
                SELECT MAX(flood_risk)
                FROM candidate_locations;
            """)
        ).scalar()

    print("\n" + "=" * 60)
    print("Flood-risk feature completed!")
    print("=" * 60)

    print("Total candidates :", total)
    print("Zero-risk/placeholder values :", zero_count)
    print("Minimum flood_risk :", minimum)
    print("Maximum flood_risk :", maximum)

    print("=" * 60)


if __name__ == "__main__":
    assign_flood_risk()