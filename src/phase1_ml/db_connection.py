import psycopg2

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5432,
    "database": "geosense_db",
    "user": "postgres",
    "password": "1011"
}


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


if __name__ == "__main__":
    try:
        conn = get_connection()
        print("PostgreSQL connection successful!")
        conn.close()
    except Exception as e:
        print("PostgreSQL connection failed:")
        print(e)