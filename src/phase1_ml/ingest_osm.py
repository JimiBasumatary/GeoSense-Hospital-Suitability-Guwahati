import geopandas as gpd
from db_connection import get_engine

# ---------------------------------------------------------
# GeoSense - Exercise 4
# Guwahati OSM Data Ingestion
# ---------------------------------------------------------

LAYERS = [
    {
        "file": r"data\shapefiles\osm\guwahati\guwahati_roads.gpkg",
        "layer": "guwahati_roads",
        "table": "guwahati_roads"
    },
    {
        "file": r"data\shapefiles\osm\guwahati\guwahati_buildings.gpkg",
        "layer": "guwahati_buildings",
        "table": "guwahati_buildings"
    },
    {
        "file": r"data\shapefiles\osm\guwahati\guwahati_landuse.gpkg",
        "layer": "guwahati_landuse",
        "table": "guwahati_landuse"
    },
    {
        "file": r"data\shapefiles\osm\guwahati\guwahati_pois.gpkg",
        "layer": "guwahati_pois",
        "table": "guwahati_pois"
    }
]


def load_layer(file_path, layer_name, table_name, engine):

    print("\n" + "=" * 60)
    print(f"Loading: {layer_name}")
    print("=" * 60)

    # Read GeoPackage
    gdf = gpd.read_file(
        file_path,
        layer=layer_name
    )

    print("Features:", len(gdf))
    print("Original CRS:", gdf.crs)
    print("Geometry types:")
    print(gdf.geom_type.value_counts().to_dict())

    # Convert to WGS 84
    gdf = gdf.to_crs(epsg=4326)

    print("Final CRS:", gdf.crs)

    # Load into PostGIS
    gdf.to_postgis(
        name=table_name,
        con=engine,
        if_exists="replace",
        index=False
    )

    print(f"SUCCESS: {table_name} loaded into PostGIS.")


def main():

    engine = get_engine()

    print("\nGeoSense OSM ingestion started...")

    for item in LAYERS:

        load_layer(
            item["file"],
            item["layer"],
            item["table"],
            engine
        )

    print("\n" + "=" * 60)
    print("ALL OSM LAYERS LOADED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()