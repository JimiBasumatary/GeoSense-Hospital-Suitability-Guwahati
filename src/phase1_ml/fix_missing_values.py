from db_connection import get_engine
from sqlalchemy import text


def fix_missing_values():

    print("\n" + "=" * 60)
    print("GeoSense - Convert NaN to SQL NULL")
    print("=" * 60)

    engine = get_engine()

    with engine.begin() as connection:

        # Population NaN -> SQL NULL
        connection.execute(text("""
            UPDATE candidate_locations
            SET population_value = NULL
            WHERE population_value::text = 'NaN';
        """))

        # Elevation NaN -> SQL NULL
        connection.execute(text("""
            UPDATE candidate_locations
            SET elevation_m = NULL
            WHERE elevation_m::text = 'NaN';
        """))

    # Verify
    with engine.connect() as connection:

        population_null = connection.execute(
            text("""
                SELECT COUNT(*)
                FROM candidate_locations
                WHERE population_value IS NULL;
            """)
        ).scalar()

        elevation_null = connection.execute(
            text("""
                SELECT COUNT(*)
                FROM candidate_locations
                WHERE elevation_m IS NULL;
            """)
        ).scalar()

        population_nan = connection.execute(
            text("""
                SELECT COUNT(*)
                FROM candidate_locations
                WHERE population_value::text = 'NaN';
            """)
        ).scalar()

        elevation_nan = connection.execute(
            text("""
                SELECT COUNT(*)
                FROM candidate_locations
                WHERE elevation_m::text = 'NaN';
            """)
        ).scalar()

    print("\n" + "=" * 60)
    print("Correction completed!")
    print("=" * 60)

    print("Population NULL :", population_null)
    print("Population NaN  :", population_nan)

    print("Elevation NULL  :", elevation_null)
    print("Elevation NaN   :", elevation_nan)

    print("=" * 60)


if __name__ == "__main__":
    fix_missing_values()