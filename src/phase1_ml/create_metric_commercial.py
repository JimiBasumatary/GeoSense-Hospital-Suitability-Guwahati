from db_connection import get_engine
from sqlalchemy import text


def create_metric_commercial():

    print("\n" + "=" * 60)
    print("GeoSense - Create Metric Commercial Land Use")
    print("=" * 60)

    engine = get_engine()

    sql = """
    DROP TABLE IF EXISTS commercial_landuse_utm;

    CREATE TABLE commercial_landuse_utm AS
    SELECT
        osm_id,
        name,
        fclass,
        ST_Transform(geometry, 32646) AS geometry
    FROM guwahati_landuse
    WHERE LOWER(fclass) = 'commercial';

    CREATE INDEX commercial_landuse_utm_geom_idx
    ON commercial_landuse_utm
    USING GIST (geometry);
    """

    with engine.begin() as connection:
        connection.execute(text(sql))

    with engine.connect() as connection:

        count = connection.execute(
            text("SELECT COUNT(*) FROM commercial_landuse_utm")
        ).scalar()

    print("\nMetric commercial land-use table created successfully.")
    print("Commercial polygons:", count)
    print("CRS: EPSG:32646")

    print("=" * 60)


if __name__ == "__main__":
    create_metric_commercial()