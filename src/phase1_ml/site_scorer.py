import os
import sys
import joblib
import pandas as pd
import numpy as np
import rasterio

from sqlalchemy import text
from pyproj import Transformer

# Allow importing db_connection.py from the same project folder
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from db_connection import get_engine


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = r"D:\GeoSense_Agent"

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "saved",
    "hospital_suitability_xgboost.joblib"
)

POPULATION_RASTER = os.path.join(
    PROJECT_ROOT,
    "data",
    "population",
    "processed",
    "population_guwahati_raw_2024.tif"
)

DEM_RASTER = os.path.join(
    PROJECT_ROOT,
    "data",
    "elevation",
    "guwahati_srtm_30m.tif"
)


# ============================================================
# LAND-USE CLASSES
# ============================================================

LANDUSE_CLASSES = [
    "cemetery",
    "commercial",
    "farmland",
    "forest",
    "grass",
    "industrial",
    "meadow",
    "military",
    "orchard",
    "park",
    "quarry",
    "recreation_ground",
    "residential",
    "retail",
    "unclassified"
]


# ============================================================
# COORDINATE TRANSFORMATION
# ============================================================

TRANSFORMER = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:32646",
    always_xy=True
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    saved = joblib.load(MODEL_PATH)

    return (
        saved["model"],
        saved["features"],
        saved["feature_medians"],
        saved["class_labels"]
    )


# ============================================================
# VALIDATE COORDINATES
# ============================================================

def validate_coordinates(latitude, longitude):

    if not (-90 <= latitude <= 90):
        raise ValueError("Latitude must be between -90 and 90.")

    if not (-180 <= longitude <= 180):
        raise ValueError("Longitude must be between -180 and 180.")


# ============================================================
# CALCULATE ROAD DISTANCE
# ============================================================

def get_road_distance(engine, point_utm):

    sql = """
        SELECT
            ST_Distance(
                :point,
                geometry
            ) AS distance_m
        FROM guwahati_roads_utm
        ORDER BY geometry <-> :point
        LIMIT 1;
    """

    with engine.connect() as connection:

        result = connection.execute(
            text(sql),
            {"point": point_utm}
        ).fetchone()

    if result is None:
        raise RuntimeError(
            "Unable to calculate distance to nearest road."
        )

    return float(result.distance_m)


# ============================================================
# CALCULATE HOSPITAL DISTANCE
# ============================================================

def get_hospital_distance(engine, point_utm):

    sql = """
        SELECT
            ST_Distance(
                :point,
                geometry
            ) AS distance_m
        FROM hospital_locations_utm
        ORDER BY geometry <-> :point
        LIMIT 1;
    """

    with engine.connect() as connection:

        result = connection.execute(
            text(sql),
            {"point": point_utm}
        ).fetchone()

    if result is None:
        raise RuntimeError(
            "Unable to calculate distance to nearest hospital."
        )

    return float(result.distance_m)


# ============================================================
# CALCULATE COMMERCIAL DISTANCE
# ============================================================

def get_commercial_distance(engine, point_utm):

    sql = """
        SELECT
            ST_Distance(
                :point,
                geometry
            ) AS distance_m
        FROM commercial_landuse_utm
        ORDER BY geometry <-> :point
        LIMIT 1;
    """

    with engine.connect() as connection:

        result = connection.execute(
            text(sql),
            {"point": point_utm}
        ).fetchone()

    if result is None:
        raise RuntimeError(
            "Unable to calculate distance to commercial land use."
        )

    return float(result.distance_m)


# ============================================================
# GET LAND-USE CLASS
# ============================================================

def get_landuse_class(engine, point_utm):

    sql = """
        SELECT
            COALESCE(
                (
                    SELECT l.fclass
                    FROM guwahati_landuse_utm AS l
                    WHERE ST_Intersects(l.geometry, :point)
                    LIMIT 1
                ),
                (
                    SELECT l.fclass
                    FROM guwahati_landuse_utm AS l
                    WHERE ST_DWithin(
                        l.geometry,
                        :point,
                        500
                    )
                    ORDER BY l.geometry <-> :point
                    LIMIT 1
                ),
                'unclassified'
            ) AS land_use_class;
    """

    with engine.connect() as connection:

        result = connection.execute(
            text(sql),
            {"point": point_utm}
        ).fetchone()

    if result is None:
        return "unclassified"

    landuse = result.land_use_class

    if landuse is None:
        return "unclassified"

    landuse = str(landuse).lower()

    if landuse not in LANDUSE_CLASSES:
        return "unclassified"

    return landuse


# ============================================================
# EXTRACT RASTER VALUE
# ============================================================

def extract_raster_value(raster_path, longitude, latitude):

    if not os.path.exists(raster_path):
        raise FileNotFoundError(
            f"Raster not found:\n{raster_path}"
        )

    with rasterio.open(raster_path) as src:

        try:
            row, col = src.index(
                longitude,
                latitude
            )

            value = src.read(1)[row, col]

        except Exception:
            return np.nan

        if src.nodata is not None:
            if value == src.nodata:
                return np.nan

        if not np.isfinite(value):
            return np.nan

        return float(value)


# ============================================================
# CREATE MODEL FEATURES
# ============================================================

def create_feature_row(
    dist_road_m,
    dist_hospital_m,
    population_value,
    elevation_m,
    dist_landuse_commercial_m,
    land_use_class
):

    data = {
        "dist_road_m": dist_road_m,
        "dist_hospital_m": dist_hospital_m,
        "population_value": population_value,
        "elevation_m": elevation_m,
        "dist_landuse_commercial_m": dist_landuse_commercial_m
    }

    # Create all land-use one-hot variables
    for landuse in LANDUSE_CLASSES:

        column_name = f"land_use_{landuse}"

        if landuse == land_use_class:
            data[column_name] = 1
        else:
            data[column_name] = 0

    return pd.DataFrame([data])


# ============================================================
# SCORE A NEW SITE
# ============================================================

def score_site(latitude, longitude):

    validate_coordinates(
        latitude,
        longitude
    )

    print("\n" + "=" * 60)
    print("GeoSense Hospital Site Scorer")
    print("=" * 60)

    print("\nInput coordinates")
    print("Latitude :", latitude)
    print("Longitude:", longitude)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, features, medians, class_labels = load_model()

    # --------------------------------------------------------
    # Convert WGS84 coordinates to UTM
    # --------------------------------------------------------

    x, y = TRANSFORMER.transform(
        longitude,
        latitude
    )

    point_utm = f"SRID=32646;POINT({x} {y})"

    print("\nUTM coordinates")
    print("X:", round(x, 3))
    print("Y:", round(y, 3))

    # --------------------------------------------------------
    # Connect to PostGIS
    # --------------------------------------------------------

    engine = get_engine()

    # --------------------------------------------------------
    # GIS feature extraction
    # --------------------------------------------------------

    print("\nCalculating GIS features...")

    dist_road_m = get_road_distance(
        engine,
        point_utm
    )

    dist_hospital_m = get_hospital_distance(
        engine,
        point_utm
    )

    dist_commercial_m = get_commercial_distance(
        engine,
        point_utm
    )

    land_use_class = get_landuse_class(
        engine,
        point_utm
    )

    population_value = extract_raster_value(
        POPULATION_RASTER,
        longitude,
        latitude
    )

    elevation_m = extract_raster_value(
        DEM_RASTER,
        longitude,
        latitude
    )

    # --------------------------------------------------------
    # Handle missing raster values
    # --------------------------------------------------------

    if pd.isna(population_value):

        population_value = float(
            medians["population_value"]
        )

        population_source = "median imputation"

    else:

        population_source = "raster extraction"

    if pd.isna(elevation_m):

        elevation_m = float(
            medians["elevation_m"]
        )

        elevation_source = "median imputation"

    else:

        elevation_source = "DEM extraction"

    # --------------------------------------------------------
    # Create feature dataframe
    # --------------------------------------------------------

    data = create_feature_row(
        dist_road_m,
        dist_hospital_m,
        population_value,
        elevation_m,
        dist_commercial_m,
        land_use_class
    )

    # Make sure every expected feature exists
    for feature in features:

        if feature not in data.columns:
            data[feature] = 0

    # Keep exactly the training feature order
    data = data[features]

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = int(
        model.predict(data)[0]
    )

    probabilities = model.predict_proba(data)[0]

    confidence = float(
        np.max(probabilities)
    )

    predicted_class = class_labels[prediction]

    # --------------------------------------------------------
    # Display extracted features
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("EXTRACTED SITE FEATURES")
    print("-" * 60)

    print(
        "Distance to road       :",
        round(dist_road_m, 2),
        "m"
    )

    print(
        "Distance to hospital   :",
        round(dist_hospital_m, 2),
        "m"
    )

    print(
        "Population value       :",
        round(population_value, 4),
        f"({population_source})"
    )

    print(
        "Elevation              :",
        round(elevation_m, 2),
        f"m ({elevation_source})"
    )

    print(
        "Distance to commercial :",
        round(dist_commercial_m, 2),
        "m"
    )

    print(
        "Land-use class         :",
        land_use_class
    )

    # --------------------------------------------------------
    # Display prediction
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("MODEL PREDICTION")
    print("-" * 60)

    print(
        "Predicted suitability  :",
        predicted_class
    )

    print(
        "Confidence             :",
        round(confidence, 4)
    )

    print("\nClass probabilities:")

    for class_id, probability in enumerate(probabilities):

        print(
            f"  {class_labels[class_id]:<10}: "
            f"{probability:.4f}"
        )

    print("\n" + "=" * 60)

    return {
        "latitude": latitude,
        "longitude": longitude,
        "dist_road_m": round(dist_road_m, 2),
        "dist_hospital_m": round(dist_hospital_m, 2),
        "population_value": round(population_value, 4),
        "elevation_m": round(elevation_m, 2),
        "dist_landuse_commercial_m": round(
            dist_commercial_m,
            2
        ),
        "land_use_class": land_use_class,
        "prediction": prediction,
        "predicted_suitability": predicted_class,
        "confidence": round(confidence, 4)
    }


# ============================================================
# TEST THE SCORER
# ============================================================

if __name__ == "__main__":

    result = score_site(
        latitude=26.128019883114455,
        longitude=91.63553723918953
    )

    print("\nFinal result:")
    print(result)