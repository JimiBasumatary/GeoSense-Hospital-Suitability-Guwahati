import geopandas as gpd
import rasterio
from rasterio.sample import sample_gen
from db_connection import get_engine
from sqlalchemy import text


RASTER = r"data\population\processed\population_guwahati_raw_2024.tif"


def extract_population():

    print("\n" + "=" * 60)
    print("GeoSense - Extract Population Values")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Read candidate locations from PostGIS
    # ---------------------------------------------------------
    print("\nReading candidate locations from PostGIS...")

    engine = get_engine()

    candidates = gpd.read_postgis(
        "SELECT location_id, geometry FROM candidate_locations",
        engine,
        geom_col="geometry"
    )

    print("Candidate locations:", len(candidates))
    print("Candidate CRS:", candidates.crs)

    # ---------------------------------------------------------
    # 2. Open population raster
    # ---------------------------------------------------------
    print("\nOpening population raster...")

    with rasterio.open(RASTER) as src:

        print("Raster CRS:", src.crs)
        print("Raster resolution:", src.res)

        # -----------------------------------------------------
        # 3. Reproject candidate points to raster CRS
        # -----------------------------------------------------
        candidates_raster = candidates.to_crs(src.crs)

        # -----------------------------------------------------
        # 4. Extract raster values
        # -----------------------------------------------------
        print("\nExtracting population values...")

        coordinates = [
            (geom.x, geom.y)
            for geom in candidates_raster.geometry
        ]

        values = []

        for value in src.sample(coordinates):
            values.append(float(value[0]))

    # ---------------------------------------------------------
    # 5. Add values to candidate GeoDataFrame
    # ---------------------------------------------------------
    candidates["population_value"] = values

    # Convert NoData to None
    candidates.loc[
        candidates["population_value"] == -99999,
        "population_value"
    ] = None

    # ---------------------------------------------------------
    # 6. Add column to PostGIS
    # ---------------------------------------------------------
    print("\nUpdating PostGIS...")

    with engine.begin() as connection:

        connection.execute(text("""
            ALTER TABLE candidate_locations
            DROP COLUMN IF EXISTS population_value;
        """))

        connection.execute(text("""
            ALTER TABLE candidate_locations
            ADD COLUMN population_value DOUBLE PRECISION;
        """))

        for _, row in candidates.iterrows():

            connection.execute(
                text("""
                    UPDATE candidate_locations
                    SET population_value = :value
                    WHERE location_id = :location_id;
                """),
                {
                    "value": row["population_value"],
                    "location_id": int(row["location_id"])
                }
            )

    # ---------------------------------------------------------
    # 7. Print statistics
    # ---------------------------------------------------------
    valid = candidates["population_value"].dropna()

    print("\n" + "=" * 60)
    print("Population extraction completed!")
    print("=" * 60)

    print("Total candidates :", len(candidates))
    print("Valid values     :", len(valid))

    if len(valid) > 0:
        print("Minimum value    :", valid.min())
        print("Maximum value    :", valid.max())
        print("Mean value       :", valid.mean())
        print("Median value     :", valid.median())

    print("\nColumn added to PostGIS:")
    print("population_value")

    print("=" * 60)


if __name__ == "__main__":
    extract_population()