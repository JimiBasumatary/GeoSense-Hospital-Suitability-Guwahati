import joblib
import pandas as pd


MODEL_PATH = "data/outputs/exercise5_random_forest.joblib"


def score_site(
    building_area_m2,
    buildings_within_250m,
    dist_road_m,
    pois_within_500m,
    dominant_landuse,
    dominant_landuse_area_m2
):
    # Load trained model and saved feature order
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    features = saved["features"]

    # Create one-row input
    data = pd.DataFrame([{
        "building_area_m2": building_area_m2,
        "buildings_within_250m": buildings_within_250m,
        "dist_road_m": dist_road_m,
        "pois_within_500m": pois_within_500m,
        "dominant_landuse_area_m2": dominant_landuse_area_m2,
        "dominant_landuse": dominant_landuse
    }])

    # Convert land-use category to the same dummy variables
    data = pd.get_dummies(
        data,
        columns=["dominant_landuse"],
        dtype=int
    )

    # Make sure every training feature exists
    for feature in features:
        if feature not in data.columns:
            data[feature] = 0

    # Keep exactly the same feature order as training
    data = data[features]

    # Prediction
    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0][1])

    label = "Suitable" if prediction == 1 else "Not suitable"

    return {
        "prediction": prediction,
        "label": label,
        "probability": round(probability, 4)
    }


if __name__ == "__main__":
    # Example site
    result = score_site(
        building_area_m2=200,
        buildings_within_250m=12,
        dist_road_m=5,
        pois_within_500m=6,
        dominant_landuse="residential",
        dominant_landuse_area_m2=5000
    )

    print("GeoSense Site Scorer")
    print("====================")
    print("Prediction:", result["label"])
    print("Probability:", result["probability"])