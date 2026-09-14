from db_connection import get_engine
from sqlalchemy import text


def create_metric_roads():

    print("\n" + "=" * 60)
    print("GeoSense - Create Metric Road Layer")
    print("=" * 60)

    engine = get_engine()

    sql = """
    DROP TABLE IF EXISTS guwahati_roads_utm;

    CREATE TABLE guwahati_roads_utm AS
    SELECT
        osm_id,
        name,
        fclass,
        ST_Transform(geometry, 32646) AS geometry
    FROM guwahati_roads;

    CREATE INDEX guwahati_roads_utm_geom_idx
    ON guwahati_roads_utm
    USING GIST (geometry);
    """

    print("\nCreating metric road table...")

    with engine.begin() as connection:
        connection.execute(text(sql))

    print("\nSUCCESS!")
    print("Metric road table created successfully.")
    print("CRS: EPSG:32646")


if __name__ == "__main__":
    create_metric_roads()