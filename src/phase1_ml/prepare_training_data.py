import pandas as pd
from sklearn.model_selection import train_test_split


INPUT = r"data\processed\hospital_suitability_dataset.csv"

X_OUTPUT = r"data\processed\X_features.csv"
Y_OUTPUT = r"data\processed\y_target.csv"

RANDOM_STATE = 42


def prepare_training_data():

    print("\n" + "=" * 75)
    print("GeoSense - ML Training Dataset Preparation")
    print("=" * 75)

    # ---------------------------------------------------------
    # 1. Read suitability dataset
    # ---------------------------------------------------------
    print("\n1. Reading suitability dataset...")

    df = pd.read_csv(INPUT)

    print("Rows    :", len(df))
    print("Columns :", len(df.columns))

    # ---------------------------------------------------------
    # 2. Define feature columns
    # ---------------------------------------------------------
    feature_columns = [
        "dist_road_m",
        "dist_hospital_m",
        "flood_risk",
        "population_value",
        "elevation_m",
        "dist_landuse_commercial_m"
    ]

    # Add one-hot encoded land-use columns
    landuse_columns = [
        column
        for column in df.columns
        if column.startswith("land_use_")
    ]

    feature_columns.extend(landuse_columns)

    # ---------------------------------------------------------
    # 3. Define X and y
    # ---------------------------------------------------------
    X = df[feature_columns].copy()

    y = df["suitability_class"].copy()

    print("\n" + "-" * 75)
    print("2. Feature and Target Definition")
    print("-" * 75)

    print("\nNumber of features:", len(feature_columns))

    print("\nFeatures:")

    for feature in feature_columns:
        print("  ", feature)

    print("\nTarget:")
    print("  suitability_class")

    # ---------------------------------------------------------
    # 4. Check missing values
    # ---------------------------------------------------------
    print("\n" + "-" * 75)
    print("3. Missing Value Check")
    print("-" * 75)

    print("\nMissing values in X:")
    print(X.isna().sum().to_string())

    print("\nMissing values in y:")
    print(y.isna().sum())

    # ---------------------------------------------------------
    # 5. Target distribution
    # ---------------------------------------------------------
    print("\n" + "-" * 75)
    print("4. Target Class Distribution")
    print("-" * 75)

    print(y.value_counts().to_string())

    # ---------------------------------------------------------
    # 6. Train-test split
    # ---------------------------------------------------------
    print("\n" + "-" * 75)
    print("5. Train-Test Split")
    print("-" * 75)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("Training samples:", len(X_train))
    print("Testing samples :", len(X_test))

    print("\nTraining percentage: 80%")
    print("Testing percentage : 20%")

    # ---------------------------------------------------------
    # 7. Save complete X and y
    # ---------------------------------------------------------
    X.to_csv(X_OUTPUT, index=False)
    y.to_csv(Y_OUTPUT, index=False)

    # ---------------------------------------------------------
    # 8. Final information
    # ---------------------------------------------------------
    print("\n" + "=" * 75)
    print("ML TRAINING DATA PREPARATION COMPLETED")
    print("=" * 75)

    print("\nFeature dataset:")
    print(X_OUTPUT)

    print("\nTarget dataset:")
    print(Y_OUTPUT)

    print("\nX shape:", X.shape)
    print("y shape:", y.shape)

    print("\nTrain shape:", X_train.shape)
    print("Test shape :", X_test.shape)

    print("\nRandom state:", RANDOM_STATE)

    print("\nNext stage:")
    print("Train and compare Machine Learning classification models.")

    print("=" * 75)


if __name__ == "__main__":
    prepare_training_data()