from db_connection import get_engine
from sqlalchemy import text


def create_metric_landuse():

    print("\n" + "=" * 60)
    print("GeoSense - Create Metric Land Use Layer")
    print("=" * 60)

    engine = get_engine()

    sql = """
    DROP TABLE IF EXISTS guwahati_landuse_utm;

    CREATE TABLE guwahati_landuse_utm AS
    SELECT
        osm_id,
        name,
        fclass,
        ST_Transform(geometry, 32646) AS geometry
    FROM guwahati_landuse;

    CREATE INDEX guwahati_landuse_utm_geom_idx
    ON guwahati_landuse_utm
    USING GIST (geometry);
    """

    with engine.begin() as connection:
        connection.execute(text(sql))

    with engine.connect() as connection:

        total = connection.execute(
            text("""
                SELECT COUNT(*)
                FROM guwahati_landuse_utm;
            """)
        ).scalar()

    print("\nMetric land-use table created successfully.")
    print("Total land-use polygons:", total)
    print("CRS: EPSG:32646")

    print("=" * 60)


if __name__ == "__main__":
    create_metric_landuse()