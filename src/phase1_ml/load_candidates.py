import geopandas as gpd
from db_connection import get_engine


INPUT_FILE = r"data\processed\candidate_locations.gpkg"
LAYER_NAME = "candidate_locations"
TABLE_NAME = "candidate_locations"


def load_candidates():

    print("\n" + "=" * 60)
    print("GeoSense - Candidate Locations → PostGIS")
    print("=" * 60)

    print("\nReading candidate locations...")

    gdf = gpd.read_file(
        INPUT_FILE,
        layer=LAYER_NAME
    )

    print("Number of candidates:", len(gdf))
    print("CRS:", gdf.crs)
    print(
        "Geometry:",
        gdf.geom_type.value_counts().to_dict()
    )

    # Make sure candidates use EPSG:32646
    gdf = gdf.to_crs(epsg=32646)

    print("Final CRS:", gdf.crs)

    print("\nLoading candidates into PostGIS...")

    engine = get_engine()

    gdf.to_postgis(
        name=TABLE_NAME,
        con=engine,
        if_exists="replace",
        index=False
    )

    print("\nSUCCESS!")
    print(
        "candidate_locations table created in PostGIS."
    )

    print("\n" + "=" * 60)
    print("CANDIDATE LOADING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    load_candidates()