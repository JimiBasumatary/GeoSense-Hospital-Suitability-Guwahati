import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


INPUT = r"data\processed\hospital_suitability_predictions.csv"

TABLE_NAME = "hospital_suitability_predictions"


def load_predictions():

    print("\n" + "=" * 75)
    print("GeoSense - Load ML Predictions into PostGIS")
    print("=" * 75)

    # ---------------------------------------------------------
    # 1. Load database configuration
    # ---------------------------------------------------------
    print("\n1. Loading database configuration...")

    load_dotenv()

    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")

    connection_string = (
        f"postgresql+psycopg2://"
        f"{DB_USER}:{DB_PASSWORD}@"
        f"{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    engine = create_engine(connection_string)

    # ---------------------------------------------------------
    # 2. Read prediction CSV
    # ---------------------------------------------------------
    print("\n2. Reading prediction CSV...")

    df = pd.read_csv(INPUT)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    # ---------------------------------------------------------
    # 3. Load predictions into temporary table
    # ---------------------------------------------------------
    print("\n3. Loading prediction attributes...")

    temp_table = "temp_hospital_predictions"

    df.to_sql(
        temp_table,
        engine,
        if_exists="replace",
        index=False
    )

    # ---------------------------------------------------------
    # 4. Create final spatial table
    # ---------------------------------------------------------
    print("\n4. Creating spatial prediction table...")

    with engine.begin() as conn:

        conn.execute(
            text(
                f"DROP TABLE IF EXISTS {TABLE_NAME}"
            )
        )

        conn.execute(
            text(
                f"""
                CREATE TABLE {TABLE_NAME} AS
                SELECT
                    c.location_id,
                    c.longitude,
                    c.latitude,
                    p.suitability_score,
                    p.suitability_class,
                    p.predicted_suitability,
                    p.prediction_confidence,
                    c.geometry
                FROM candidate_locations c
                JOIN {temp_table} p
                ON c.location_id = p.location_id
                """
            )
        )

        # -----------------------------------------------------
        # 5. Create spatial index
        # -----------------------------------------------------
        conn.execute(
            text(
                f"""
                CREATE INDEX
                idx_{TABLE_NAME}_geometry
                ON {TABLE_NAME}
                USING GIST (geometry)
                """
            )
        )

        # -----------------------------------------------------
        # 6. Create primary key
        # -----------------------------------------------------
        conn.execute(
            text(
                f"""
                ALTER TABLE {TABLE_NAME}
                ADD PRIMARY KEY (location_id)
                """
            )
        )

        # -----------------------------------------------------
        # 7. Remove temporary table
        # -----------------------------------------------------
        conn.execute(
            text(
                f"DROP TABLE IF EXISTS {temp_table}"
            )
        )

    # ---------------------------------------------------------
    # 8. Verify result
    # ---------------------------------------------------------
    print("\n5. Verifying PostGIS layer...")

    with engine.connect() as conn:

        total = conn.execute(
            text(
                f"""
                SELECT COUNT(*)
                FROM {TABLE_NAME}
                """
            )
        ).scalar()

        classes = conn.execute(
            text(
                f"""
                SELECT
                    predicted_suitability,
                    COUNT(*)
                FROM {TABLE_NAME}
                GROUP BY predicted_suitability
                ORDER BY predicted_suitability
                """
            )
        ).fetchall()

    print("\nTotal spatial predictions:", total)

    print("\nPredicted suitability:")

    for row in classes:
        print(" ", row[0], ":", row[1])

    # ---------------------------------------------------------
    # 9. Final message
    # ---------------------------------------------------------
    print("\n" + "=" * 75)
    print("ML PREDICTIONS LOADED INTO POSTGIS")
    print("=" * 75)

    print("\nPostGIS table:")
    print(TABLE_NAME)

    print("\nRecords:", total)

    print("\nNext stage:")
    print("Export the spatial prediction layer for GIS visualization.")

    print("=" * 75)


if __name__ == "__main__":
    load_predictions()