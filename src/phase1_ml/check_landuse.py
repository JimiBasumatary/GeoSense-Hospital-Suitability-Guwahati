from db_connection import get_engine
from sqlalchemy import text

engine = get_engine()

with engine.connect() as connection:

    total = connection.execute(
        text("SELECT COUNT(*) FROM guwahati_landuse")
    ).scalar()

    commercial = connection.execute(
        text("""
            SELECT COUNT(*)
            FROM guwahati_landuse
            WHERE LOWER(fclass) = 'commercial'
        """)
    ).scalar()

    print("\n" + "=" * 60)
    print("Land Use Check")
    print("=" * 60)

    print("Total land-use features :", total)
    print("Commercial features     :", commercial)

    print("=" * 60)