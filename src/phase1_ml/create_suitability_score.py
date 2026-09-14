import pandas as pd
import numpy as np

INPUT = r"data\processed\features_encoded.csv"
OUTPUT = r"data\processed\hospital_suitability_dataset.csv"


def min_max_score(series):
    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(1.0, index=series.index)

    return (series - minimum) / (maximum - minimum)


def create_suitability_score():

    print("\n" + "=" * 75)
    print("GeoSense - Hospital Suitability Score Creation")
    print("=" * 75)

    # ---------------------------------------------------------
    # 1. Read encoded dataset
    # ---------------------------------------------------------
    print("\n1. Reading encoded dataset...")

    df = pd.read_csv(INPUT)

    print("Rows    :", len(df))
    print("Columns :", len(df.columns))

    # ---------------------------------------------------------
    # 2. Numerical criterion scores
    # ---------------------------------------------------------
    print("\n2. Calculating criterion scores...")

    # Distance to road:
    # closer to roads = better
    road_normalized = min_max_score(df["dist_road_m"])
    df["score_road"] = 1 - road_normalized

    # Population:
    # higher population = better
    df["score_population"] = min_max_score(
        df["population_value"]
    )

    # Distance to existing hospitals:
    # greater distance = greater potential service gap
    df["score_hospital_gap"] = min_max_score(
        df["dist_hospital_m"]
    )

    # Elevation:
    # higher elevation = generally preferred
    df["score_elevation"] = min_max_score(
        df["elevation_m"]
    )

    # Distance to commercial land use:
    # closer = better accessibility
    commercial_normalized = min_max_score(
        df["dist_landuse_commercial_m"]
    )

    df["score_commercial"] = 1 - commercial_normalized

    # ---------------------------------------------------------
    # 3. Land-use suitability
    # ---------------------------------------------------------
    print("\n3. Assigning land-use suitability scores...")

    landuse_scores = {
        "commercial": 1.0,
        "retail": 1.0,
        "residential": 0.8,
        "farmland": 0.6,
        "grass": 0.6,
        "meadow": 0.6,
        "park": 0.5,
        "recreation_ground": 0.5,
        "forest": 0.3,
        "orchard": 0.3,
        "industrial": 0.2,
        "unclassified": 0.3,
        "quarry": 0.0,
        "cemetery": 0.0,
        "military": 0.0
    }

    # Recover land-use from one-hot columns
    landuse_columns = [
        col for col in df.columns
        if col.startswith("land_use_")
    ]

    def get_landuse_score(row):

        for column in landuse_columns:

            if row[column] == 1:

                landuse_class = column.replace(
                    "land_use_", ""
                )

                return landuse_scores.get(
                    landuse_class,
                    0.3
                )

        return 0.3

    df["score_landuse"] = df.apply(
        get_landuse_score,
        axis=1
    )

    # ---------------------------------------------------------
    # 4. Weighted suitability score
    # ---------------------------------------------------------
    print("\n4. Calculating weighted suitability score...")

    WEIGHTS = {
        "score_road": 0.20,
        "score_population": 0.25,
        "score_hospital_gap": 0.20,
        "score_elevation": 0.15,
        "score_commercial": 0.10,
        "score_landuse": 0.10
    }

    df["suitability_score"] = (
        df["score_road"] * WEIGHTS["score_road"]
        + df["score_population"] * WEIGHTS["score_population"]
        + df["score_hospital_gap"] * WEIGHTS["score_hospital_gap"]
        + df["score_elevation"] * WEIGHTS["score_elevation"]
        + df["score_commercial"] * WEIGHTS["score_commercial"]
        + df["score_landuse"] * WEIGHTS["score_landuse"]
    )

    # ---------------------------------------------------------
    # 5. Create suitability classes
    # ---------------------------------------------------------
    print("\n5. Creating suitability classes...")

    df["suitability_class"] = pd.qcut(
    df["suitability_score"],
    q=3,
    labels=[
        "Low",
        "Moderate",
        "High"
    ]
)

    # ---------------------------------------------------------
    # 6. Display results
    # ---------------------------------------------------------
    print("\n" + "-" * 75)
    print("6. Suitability Score Statistics")
    print("-" * 75)

    print(
        df["suitability_score"]
        .describe()
        .round(4)
        .to_string()
    )

    print("\nSuitability classes:")

    print(
        df["suitability_class"]
        .value_counts()
        .to_string()
    )

    # ---------------------------------------------------------
    # 7. Check score range
    # ---------------------------------------------------------
    print("\n" + "-" * 75)
    print("7. Score Range")
    print("-" * 75)

    print(
        "Minimum score:",
        round(df["suitability_score"].min(), 4)
    )

    print(
        "Maximum score:",
        round(df["suitability_score"].max(), 4)
    )

    # ---------------------------------------------------------
    # 8. Save dataset
    # ---------------------------------------------------------
    df.to_csv(
        OUTPUT,
        index=False
    )

    print("\n" + "=" * 75)
    print("HOSPITAL SUITABILITY DATASET CREATED")
    print("=" * 75)

    print("\nRows:", len(df))
    print("Columns:", len(df.columns))

    print("\nOutput:")
    print(OUTPUT)

    print("\nFlood-risk note:")
    print(
        "flood_risk was excluded because the current "
        "values are placeholders (all 0)."
    )

    print("\nNext stage:")
    print("Prepare the target and ML training dataset.")

    print("=" * 75)


if __name__ == "__main__":
    create_suitability_score()