import geopandas as gpd
import rasterio
from db_connection import get_engine
from sqlalchemy import text


RASTER = r"data\elevation\guwahati_srtm_30m.tif"


def extract_elevation():

    print("\n" + "=" * 60)
    print("GeoSense - Extract Elevation Values")
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
    # 2. Open SRTM DEM
    # ---------------------------------------------------------
    print("\nOpening SRTM DEM...")

    with rasterio.open(RASTER) as src:

        print("DEM CRS:", src.crs)
        print("DEM resolution:", src.res)
        print("DEM NoData:", src.nodata)

        # -----------------------------------------------------
        # 3. Make sure candidate points use DEM CRS
        # -----------------------------------------------------
        candidates_dem = candidates.to_crs(src.crs)

        # -----------------------------------------------------
        # 4. Extract elevation
        # -----------------------------------------------------
        print("\nExtracting elevation values...")

        coordinates = [
            (geom.x, geom.y)
            for geom in candidates_dem.geometry
        ]

        values = []

        for value in src.sample(coordinates):
            values.append(float(value[0]))

    # ---------------------------------------------------------
    # 5. Add elevation values
    # ---------------------------------------------------------
    candidates["elevation_m"] = values

    # Convert DEM NoData to NULL
    candidates.loc[
        candidates["elevation_m"] == 32767,
        "elevation_m"
    ] = None

    # ---------------------------------------------------------
    # 6. Update PostGIS
    # ---------------------------------------------------------
    print("\nUpdating PostGIS...")

    with engine.begin() as connection:

        connection.execute(text("""
            ALTER TABLE candidate_locations
            DROP COLUMN IF EXISTS elevation_m;
        """))

        connection.execute(text("""
            ALTER TABLE candidate_locations
            ADD COLUMN elevation_m DOUBLE PRECISION;
        """))

        for _, row in candidates.iterrows():

            connection.execute(
                text("""
                    UPDATE candidate_locations
                    SET elevation_m = :value
                    WHERE location_id = :location_id;
                """),
                {
                    "value": row["elevation_m"],
                    "location_id": int(row["location_id"])
                }
            )

    # ---------------------------------------------------------
    # 7. Statistics
    # ---------------------------------------------------------
    valid = candidates["elevation_m"].dropna()

    print("\n" + "=" * 60)
    print("Elevation extraction completed!")
    print("=" * 60)

    print("Total candidates :", len(candidates))
    print("Valid values     :", len(valid))

    if len(valid) > 0:
        print("Minimum elevation:", valid.min(), "m")
        print("Maximum elevation:", valid.max(), "m")
        print("Mean elevation   :", valid.mean(), "m")
        print("Median elevation :", valid.median(), "m")

    print("\nColumn added to PostGIS:")
    print("elevation_m")

    print("=" * 60)


if __name__ == "__main__":
    extract_elevation()