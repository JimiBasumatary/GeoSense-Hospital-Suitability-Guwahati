import pandas as pd


FILE = r"data\processed\features.csv"


def quality_check():

    print("\n" + "=" * 70)
    print("GeoSense - Final Feature Dataset Quality Check")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Read CSV
    # ---------------------------------------------------------

    df = pd.read_csv(FILE)

    print("\nDataset shape:")
    print("Rows    :", df.shape[0])
    print("Columns :", df.shape[1])

    # ---------------------------------------------------------
    # 2. Data types
    # ---------------------------------------------------------

    print("\nData types:")

    print(df.dtypes)

    # ---------------------------------------------------------
    # 3. Missing values
    # ---------------------------------------------------------

    print("\nMissing values:")

    print(df.isna().sum())

    # ---------------------------------------------------------
    # 4. Numeric statistics
    # ---------------------------------------------------------

    numeric_columns = [
        "dist_road_m",
        "dist_hospital_m",
        "flood_risk",
        "population_value",
        "elevation_m",
        "dist_landuse_commercial_m"
    ]

    print("\nNumeric feature statistics:")

    print(
        df[numeric_columns].describe().round(2)
    )

    # ---------------------------------------------------------
    # 5. Land-use distribution
    # ---------------------------------------------------------

    print("\nLand-use class distribution:")

    print(
        df["land_use_class"]
        .fillna("unclassified")
        .value_counts()
    )

    # ---------------------------------------------------------
    # 6. Duplicate location IDs
    # ---------------------------------------------------------

    duplicates = df["location_id"].duplicated().sum()

    print("\nDuplicate location IDs:", duplicates)

    # ---------------------------------------------------------
    # 7. Duplicate coordinates
    # ---------------------------------------------------------

    duplicate_coordinates = df.duplicated(
        subset=["longitude", "latitude"]
    ).sum()

    print(
        "Duplicate coordinate pairs:",
        duplicate_coordinates
    )

    print("\n" + "=" * 70)
    print("Quality check completed.")
    print("=" * 70)


if __name__ == "__main__":
    quality_check()