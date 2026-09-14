import pandas as pd

from sklearn.ensemble import RandomForestClassifier


INPUT = r"data\processed\hospital_suitability_dataset.csv"
OUTPUT = r"data\processed\hospital_suitability_predictions.csv"

RANDOM_STATE = 42


def generate_predictions():

    print("\n" + "=" * 75)
    print("GeoSense - Final Hospital Suitability Prediction")
    print("=" * 75)

    # ---------------------------------------------------------
    # 1. Read suitability dataset
    # ---------------------------------------------------------
    print("\n1. Reading suitability dataset...")

    df = pd.read_csv(INPUT)

    print("Rows    :", len(df))
    print("Columns :", len(df.columns))

    # ---------------------------------------------------------
    # 2. Define model features
    # ---------------------------------------------------------
    print("\n2. Preparing model features...")

    feature_columns = [
        "dist_road_m",
        "dist_hospital_m",
        "population_value",
        "elevation_m",
        "dist_landuse_commercial_m"
    ]

    # Add one-hot land-use variables
    landuse_columns = [
        column
        for column in df.columns
        if column.startswith("land_use_")
    ]

    feature_columns.extend(landuse_columns)

    X = df[feature_columns]
    y = df["suitability_class"]

    print("Number of features:", len(feature_columns))

    # ---------------------------------------------------------
    # 3. Train final Random Forest
    # ---------------------------------------------------------
    print("\n3. Training final Random Forest...")

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(X, y)

    print("Training completed using all", len(df), "locations.")

    # ---------------------------------------------------------
    # 4. Generate predictions
    # ---------------------------------------------------------
    print("\n4. Generating suitability predictions...")

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)

    # Maximum class probability
    prediction_confidence = probabilities.max(axis=1)

    df["predicted_suitability"] = predictions
    df["prediction_confidence"] = prediction_confidence

    # ---------------------------------------------------------
    # 5. Save predictions
    # ---------------------------------------------------------
    print("\n5. Saving prediction results...")

    output_columns = [
        "location_id",
        "longitude",
        "latitude",
        "suitability_score",
        "suitability_class",
        "predicted_suitability",
        "prediction_confidence"
    ]

    result = df[output_columns].copy()

    result.to_csv(
        OUTPUT,
        index=False
    )

    # ---------------------------------------------------------
    # 6. Prediction distribution
    # ---------------------------------------------------------
    print("\n" + "-" * 75)
    print("6. Predicted Suitability Distribution")
    print("-" * 75)

    print(
        result["predicted_suitability"]
        .value_counts()
        .to_string()
    )

    # ---------------------------------------------------------
    # 7. Compare baseline and predicted classes
    # ---------------------------------------------------------
    print("\n" + "-" * 75)
    print("7. Baseline vs Predicted")
    print("-" * 75)

    comparison = pd.crosstab(
        result["suitability_class"],
        result["predicted_suitability"]
    )

    print(comparison)

    # ---------------------------------------------------------
    # 8. Confidence statistics
    # ---------------------------------------------------------
    print("\n" + "-" * 75)
    print("8. Prediction Confidence")
    print("-" * 75)

    print(
        result["prediction_confidence"]
        .describe()
        .round(4)
        .to_string()
    )

    # ---------------------------------------------------------
    # 9. Highest suitability locations
    # ---------------------------------------------------------
    print("\n" + "-" * 75)
    print("9. Top 10 Candidate Locations")
    print("-" * 75)

    top10 = result.sort_values(
        "suitability_score",
        ascending=False
    ).head(10)

    print(
        top10.to_string(index=False)
    )

    # ---------------------------------------------------------
    # 10. Final message
    # ---------------------------------------------------------
    print("\n" + "=" * 75)
    print("FINAL SUITABILITY PREDICTION COMPLETED")
    print("=" * 75)

    print("\nTotal candidate locations:", len(result))

    print("\nOutput:")
    print(OUTPUT)

    print("\nModel:")
    print("Random Forest Classifier")

    print("\nNumber of trees:")
    print("300")

    print("\nNext stage:")
    print("Join predictions to the candidate GIS layer and")
    print("create the final hospital suitability map.")

    print("=" * 75)


if __name__ == "__main__":
    generate_predictions()