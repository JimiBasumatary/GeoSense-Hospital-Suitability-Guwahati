from db_connection import get_engine
from sqlalchemy import text


def calculate_hospital_distance():

    print("\n" + "=" * 60)
    print("GeoSense - Hospital Distance Feature")
    print("=" * 60)

    engine = get_engine()

    sql = """
    ALTER TABLE candidate_locations
    DROP COLUMN IF EXISTS dist_hospital_m;

    ALTER TABLE candidate_locations
    ADD COLUMN dist_hospital_m DOUBLE PRECISION;

    UPDATE candidate_locations AS c
    SET dist_hospital_m = nearest.distance_m
    FROM (
        SELECT
            c2.location_id,
            ST_Distance(c2.geometry, h.geometry) AS distance_m
        FROM candidate_locations AS c2
        CROSS JOIN LATERAL (
            SELECT geometry
            FROM hospital_locations_utm AS h
            ORDER BY h.geometry <-> c2.geometry
            LIMIT 1
        ) AS h
    ) AS nearest
    WHERE c.location_id = nearest.location_id;
    """

    print("\nCalculating distance to nearest hospital...")

    with engine.begin() as connection:
        connection.execute(text(sql))

    print("\nSUCCESS!")
    print("dist_hospital_m calculated for all candidate locations.")

    check_sql = """
    SELECT
        COUNT(*) AS total_candidates,
        COUNT(dist_hospital_m) AS calculated_distances,
        ROUND(MIN(dist_hospital_m)::numeric, 2) AS min_distance_m,
        ROUND(MAX(dist_hospital_m)::numeric, 2) AS max_distance_m,
        ROUND(AVG(dist_hospital_m)::numeric, 2) AS mean_distance_m
    FROM candidate_locations;
    """

    with engine.connect() as connection:
        result = connection.execute(text(check_sql)).fetchone()

        print("\n" + "-" * 60)
        print("HOSPITAL DISTANCE SUMMARY")
        print("-" * 60)

        print("Total candidates :", result.total_candidates)
        print("Calculated       :", result.calculated_distances)
        print("Minimum distance :", result.min_distance_m, "m")
        print("Maximum distance :", result.max_distance_m, "m")
        print("Mean distance    :", result.mean_distance_m, "m")


if __name__ == "__main__":
    calculate_hospital_distance()