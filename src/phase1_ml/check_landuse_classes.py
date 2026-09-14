from db_connection import get_engine
from sqlalchemy import text


engine = get_engine()

with engine.connect() as connection:

    rows = connection.execute(
        text("""
            SELECT
                fclass,
                COUNT(*) AS feature_count
            FROM guwahati_landuse
            GROUP BY fclass
            ORDER BY feature_count DESC;
        """)
    ).fetchall()

print("\n" + "=" * 60)
print("Land Use Classes")
print("=" * 60)

for row in rows:
    print(f"{row.fclass:25} {row.feature_count}")

print("=" * 60)