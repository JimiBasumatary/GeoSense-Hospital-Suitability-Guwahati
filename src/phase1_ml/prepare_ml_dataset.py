import pandas as pd


INPUT = r"data\processed\features.csv"
OUTPUT = r"data\processed\features_ml_ready.csv"


def prepare_ml_dataset():

    print("\n" + "=" * 70)
    print("GeoSense - Prepare ML-Ready Feature Dataset")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Read original feature dataset
    # ---------------------------------------------------------

    print("\nReading features.csv...")

    df = pd.read_csv(INPUT)

    print("Original rows:", len(df))

    # ---------------------------------------------------------
    # 2. Handle missing population values
    # ---------------------------------------------------------

    population_median = df["population_value"].median()

    print("\nPopulation median:", population_median)

    df["population_value"] = df["population_value"].fillna(
        population_median
    )

    # ---------------------------------------------------------
    # 3. Handle missing elevation values
    # ---------------------------------------------------------

    elevation_median = df["elevation_m"].median()

    print("Elevation median:", elevation_median)

    df["elevation_m"] = df["elevation_m"].fillna(
        elevation_median
    )

    # ---------------------------------------------------------
    # 4. Handle unclassified land use
    # ---------------------------------------------------------

    df["land_use_class"] = df["land_use_class"].fillna(
        "unclassified"
    )

    # ---------------------------------------------------------
    # 5. Save ML-ready dataset
    # ---------------------------------------------------------

    df.to_csv(OUTPUT, index=False)

    # ---------------------------------------------------------
    # 6. Verification
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("ML-ready dataset created successfully!")
    print("=" * 70)

    print("Rows    :", len(df))
    print("Columns :", len(df.columns))

    print("\nRemaining missing values:")

    print(df.isna().sum())

    print("\nLand-use classes:")

    print(df["land_use_class"].value_counts())

    print("\nOutput:")
    print(OUTPUT)

    print("=" * 70)


if __name__ == "__main__":
    prepare_ml_dataset()