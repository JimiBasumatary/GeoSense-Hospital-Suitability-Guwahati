from db_connection import get_engine
from sqlalchemy import text


def check_landuse_coverage():

    print("\n" + "=" * 60)
    print("GeoSense - Check Land Use Coverage")
    print("=" * 60)

    engine = get_engine()

    with engine.connect() as connection:

        for distance in [250, 500, 1000]:

            count = connection.execute(
                text("""
                    SELECT COUNT(*)
                    FROM candidate_locations AS c
                    WHERE EXISTS (
                        SELECT 1
                        FROM guwahati_landuse_utm AS l
                        WHERE ST_DWithin(
                            c.geometry,
                            l.geometry,
                            :distance
                        )
                    );
                """),
                {"distance": distance}
            ).scalar()

            print(
                f"Candidates within {distance:4} m of land-use polygon: {count}"
            )

    print("=" * 60)


if __name__ == "__main__":
    check_landuse_coverage()