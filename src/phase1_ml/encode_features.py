import pandas as pd

INPUT = r"data\processed\features_ml_ready.csv"
OUTPUT = r"data\processed\features_encoded.csv"


def encode_features():

    print("\n" + "=" * 75)
    print("GeoSense - Feature Encoding")
    print("=" * 75)

    # ---------------------------------------------------------
    # 1. Read ML-ready dataset
    # ---------------------------------------------------------
    print("\n1. Reading ML-ready dataset...")

    df = pd.read_csv(INPUT)

    print("Rows    :", len(df))
    print("Columns :", len(df.columns))

    # ---------------------------------------------------------
    # 2. Display land-use classes
    # ---------------------------------------------------------
    print("\n" + "-" * 75)
    print("2. Land-use classes before encoding")
    print("-" * 75)

    print(df["land_use_class"].value_counts().to_string())

    # ---------------------------------------------------------
    # 3. One-hot encode land-use
    # ---------------------------------------------------------
    print("\n" + "-" * 75)
    print("3. Applying One-Hot Encoding")
    print("-" * 75)

    df_encoded = pd.get_dummies(
        df,
        columns=["land_use_class"],
        prefix="land_use",
        dtype=int
    )

    # ---------------------------------------------------------
    # 4. Save encoded dataset
    # ---------------------------------------------------------
    df_encoded.to_csv(OUTPUT, index=False)

    # ---------------------------------------------------------
    # 5. Show new columns
    # ---------------------------------------------------------
    print("\n" + "-" * 75)
    print("4. Encoded Dataset")
    print("-" * 75)

    print("Rows    :", len(df_encoded))
    print("Columns :", len(df_encoded.columns))

    print("\nEncoded land-use columns:")

    landuse_columns = [
        col for col in df_encoded.columns
        if col.startswith("land_use_")
    ]

    for column in landuse_columns:
        print("  ", column)

    # ---------------------------------------------------------
    # 6. Check missing values
    # ---------------------------------------------------------
    print("\n" + "-" * 75)
    print("5. Missing Value Check")
    print("-" * 75)

    missing = df_encoded.isna().sum()

    print(missing.to_string())

    # ---------------------------------------------------------
    # 7. Final information
    # ---------------------------------------------------------
    print("\n" + "=" * 75)
    print("FEATURE ENCODING COMPLETED SUCCESSFULLY")
    print("=" * 75)

    print("\nOriginal dataset:")
    print(INPUT)

    print("\nEncoded dataset:")
    print(OUTPUT)

    print("\nRows:", len(df_encoded))
    print("Columns:", len(df_encoded.columns))

    print("\nNext stage:")
    print("Create the suitability target/score for Machine Learning")

    print("=" * 75)


if __name__ == "__main__":
    encode_features()