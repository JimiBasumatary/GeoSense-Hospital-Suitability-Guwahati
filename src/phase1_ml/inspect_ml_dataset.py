import pandas as pd

INPUT = r"data\processed\features_ml_ready.csv"


def inspect_dataset():

    print("\n" + "=" * 75)
    print("GeoSense - ML Ready Dataset Inspection")
    print("=" * 75)

    # ---------------------------------------------------------
    # 1. Read dataset
    # ---------------------------------------------------------
    print("\n1. Reading ML-ready dataset...")

    df = pd.read_csv(INPUT)

    print("Rows    :", len(df))
    print("Columns :", len(df))

    # ---------------------------------------------------------
    # 2. Column information
    # ---------------------------------------------------------
    print("\n" + "-" * 75)
    print("2. Dataset Columns")
    print("-" * 75)

    for i, column in enumerate(df.columns, start=1):
        print(f"{i:2}. {column}")

    # ---------------------------------------------------------
    # 3. Data types
    # ---------------------------------------------------------
    print("\n" + "-" * 75)
    print("3. Data Types")
    print("-" * 75)

    print(df.dtypes)

    # ---------------------------------------------------------
    # 4. Missing values
    # ---------------------------------------------------------
    print("\n" + "-" * 75)
    print("4. Missing Values")
    print("-" * 75)

    print(df.isna().sum())

    # ---------------------------------------------------------
    # 5. Numerical statistics
    # ---------------------------------------------------------
    print("\n" + "-" * 75)
    print("5. Numerical Feature Statistics")
    print("-" * 75)

    numerical_columns = [
        "dist_road_m",
        "dist_hospital_m",
        "flood_risk",
        "population_value",
        "elevation_m",
        "dist_landuse_commercial_m"
    ]

    print(df[numerical_columns].describe().round(2).to_string())

    # ---------------------------------------------------------
    # 6. Land-use classes
    # ---------------------------------------------------------
    print("\n" + "-" * 75)
    print("6. Land-use Class Distribution")
    print("-" * 75)

    print(df["land_use_class"].value_counts().to_string())

    # ---------------------------------------------------------
    # 7. Coordinate range
    # ---------------------------------------------------------
    print("\n" + "-" * 75)
    print("7. Coordinate Range")
    print("-" * 75)

    print("Longitude:")
    print("  Minimum:", df["longitude"].min())
    print("  Maximum:", df["longitude"].max())

    print("\nLatitude:")
    print("  Minimum:", df["latitude"].min())
    print("  Maximum:", df["latitude"].max())

    # ---------------------------------------------------------
    # 8. Duplicate check
    # ---------------------------------------------------------
    print("\n" + "-" * 75)
    print("8. Duplicate Check")
    print("-" * 75)

    print("Duplicate rows:",
          df.duplicated().sum())

    print("Duplicate location IDs:",
          df["location_id"].duplicated().sum())

    # ---------------------------------------------------------
    # 9. Flood risk check
    # ---------------------------------------------------------
    print("\n" + "-" * 75)
    print("9. Flood Risk Check")
    print("-" * 75)

    print(df["flood_risk"].value_counts().to_string())

    # ---------------------------------------------------------
    # 10. Final message
    # ---------------------------------------------------------
    print("\n" + "=" * 75)
    print("DATASET INSPECTION COMPLETED")
    print("=" * 75)

    print("\nML-ready dataset:")
    print(INPUT)

    print("\nNext stage:")
    print("Feature encoding and ML dataset preparation")

    print("=" * 75)


if __name__ == "__main__":
    inspect_dataset()