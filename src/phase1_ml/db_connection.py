import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# Load variables from .env
load_dotenv()


# Read database credentials
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


def get_engine():
    connection_url = (
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    engine = create_engine(connection_url)
    return engine


if __name__ == "__main__":
    try:
        engine = get_engine()

        with engine.connect() as connection:
            result = connection.execute(
                text("SELECT PostGIS_Version();")
            )

            postgis_version = result.scalar()

        print(f"Connected! PostGIS version: {postgis_version}")

    except Exception as e:
        print("Database connection failed:")
        print(e)