import geopandas as gpd
import numpy as np
from shapely.geometry import Point


# ---------------------------------------------------------
# GeoSense - Exercise 4
# Candidate Location Generation
# Study Area: Guwahati
# ---------------------------------------------------------

BOUNDARY_FILE = r"data\boundaries\guwahati\guwahati.shp"

OUTPUT_FILE = r"data\processed\candidate_locations.gpkg"

# Candidate spacing in metres
GRID_SPACING = 500


def generate_candidate_locations():

    print("\n" + "=" * 60)
    print("GeoSense - Candidate Location Generation")
    print("=" * 60)

    # -----------------------------------------------------
    # STEP 1: Read Guwahati boundary
    # -----------------------------------------------------

    print("\nSTEP 1: Reading Guwahati boundary...")

    boundary = gpd.read_file(BOUNDARY_FILE)

    print("Boundary CRS:", boundary.crs)
    print("Boundary features:", len(boundary))

    # -----------------------------------------------------
    # STEP 2: Convert to metric CRS
    # -----------------------------------------------------

    print("\nSTEP 2: Converting boundary to EPSG:32646...")

    boundary = boundary.to_crs(epsg=32646)

    print("Working CRS:", boundary.crs)

    # -----------------------------------------------------
    # STEP 3: Merge boundary geometry
    # -----------------------------------------------------

    study_area = boundary.geometry.union_all()

    print("Study-area geometry created.")

    # -----------------------------------------------------
    # STEP 4: Get boundary extent
    # -----------------------------------------------------

    minx, miny, maxx, maxy = study_area.bounds

    print("\nStudy-area bounds:")
    print("Minimum X:", minx)
    print("Minimum Y:", miny)
    print("Maximum X:", maxx)
    print("Maximum Y:", maxy)

    # -----------------------------------------------------
    # STEP 5: Create grid coordinates
    # -----------------------------------------------------

    print("\nSTEP 3: Creating candidate grid...")
    print("Grid spacing:", GRID_SPACING, "metres")

    x_values = np.arange(
        minx,
        maxx + GRID_SPACING,
        GRID_SPACING
    )

    y_values = np.arange(
        miny,
        maxy + GRID_SPACING,
        GRID_SPACING
    )

    print("Number of X positions:", len(x_values))
    print("Number of Y positions:", len(y_values))

    # -----------------------------------------------------
    # STEP 6: Generate points
    # -----------------------------------------------------

    print("\nSTEP 4: Creating grid points...")

    points = []

    for x in x_values:

        for y in y_values:

            point = Point(x, y)

            if study_area.contains(point):

                points.append(point)

    print(
        "Candidate points inside Guwahati boundary:",
        len(points)
    )

    # -----------------------------------------------------
    # STEP 7: Create GeoDataFrame
    # -----------------------------------------------------

    print("\nSTEP 5: Creating candidate GeoDataFrame...")

    candidates = gpd.GeoDataFrame(
        {
            "location_id": range(
                1,
                len(points) + 1
            )
        },
        geometry=points,
        crs="EPSG:32646"
    )

    # -----------------------------------------------------
    # STEP 8: Create latitude and longitude
    # -----------------------------------------------------

    print("\nSTEP 6: Calculating latitude and longitude...")

    candidates_wgs84 = candidates.to_crs(
        epsg=4326
    )

    candidates["longitude"] = (
        candidates_wgs84.geometry.x
    )

    candidates["latitude"] = (
        candidates_wgs84.geometry.y
    )

    # -----------------------------------------------------
    # STEP 9: Save candidate locations
    # -----------------------------------------------------

    print("\nSTEP 7: Saving candidate locations...")

    candidates.to_file(
        OUTPUT_FILE,
        layer="candidate_locations",
        driver="GPKG"
    )

    print("\nCandidate locations saved successfully.")

    print("Output:")
    print(OUTPUT_FILE)

    # -----------------------------------------------------
    # STEP 10: Display sample
    # -----------------------------------------------------

    print("\nFirst 10 candidate locations:")

    print(
        candidates[
            [
                "location_id",
                "longitude",
                "latitude"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\n" + "=" * 60)
    print("CANDIDATE LOCATION GENERATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":

    generate_candidate_locations()