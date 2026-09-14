from db_connection import get_engine
from sqlalchemy import text


def create_metric_hospitals():

    print("\n" + "=" * 60)
    print("GeoSense - Create Metric Hospital Layer")
    print("=" * 60)

    engine = get_engine()

    sql = """
    DROP TABLE IF EXISTS hospital_locations_utm;

    CREATE TABLE hospital_locations_utm AS
    SELECT
        osm_id,
        name,
        fclass,
        ST_Transform(geometry, 32646) AS geometry
    FROM hospital_locations;

    CREATE INDEX hospital_locations_utm_geom_idx
    ON hospital_locations_utm
    USING GIST (geometry);
    """

    print("\nCreating metric hospital table...")

    with engine.begin() as connection:
        connection.execute(text(sql))

    print("\nSUCCESS!")
    print("Metric hospital table created successfully.")
    print("CRS: EPSG:32646")


if __name__ == "__main__":
    create_metric_hospitals()