from db_connection import get_engine
from sqlalchemy import text


def extract_hospitals():

    engine = get_engine()

    print("\nExtracting hospitals from PostGIS...")

    sql = """
    DROP TABLE IF EXISTS hospital_locations;

    CREATE TABLE hospital_locations AS
    SELECT
        osm_id,
        name,
        fclass,
        geometry
    FROM guwahati_pois
    WHERE LOWER(fclass) = 'hospital';

    CREATE INDEX hospital_locations_geom_idx
    ON hospital_locations
    USING GIST (geometry);
    """

    with engine.begin() as connection:
        connection.execute(text(sql))

    print("Hospital table created successfully.")

    # Check number of hospitals
    with engine.connect() as connection:

        result = connection.execute(
            text("""
                SELECT COUNT(*)
                FROM hospital_locations;
            """)
        )

        count = result.scalar()

        print("Number of hospitals:", count)

        # Show hospital names
        result = connection.execute(
            text("""
                SELECT osm_id, name
                FROM hospital_locations
                ORDER BY name;
            """)
        )

        print("\nHospital locations:")
        for row in result:
            print(row[0], "|", row[1])


if __name__ == "__main__":
    extract_hospitals()