import importlib
from importlib.metadata import version, PackageNotFoundError

print("=" * 65)
print("GeoSense Environment - Library Verification")
print("=" * 65)

packages = {
    "GeoPandas": "geopandas",
    "Rasterio": "rasterio",
    "Shapely": "shapely",
    "Fiona": "fiona",
    "GDAL": "osgeo",
    "NumPy": "numpy",
    "scikit-learn": "sklearn",
    "XGBoost": "xgboost",
    "SHAP": "shap",
    "Numba": "numba",
    "psycopg2-binary": "psycopg2",
    "SQLAlchemy": "sqlalchemy",
    "GeoAlchemy2": "geoalchemy2",
    "Jupyter": "jupyter",
    "ipykernel": "ipykernel"
}

all_ok = True

print("\nLibrary verification:")
print("-" * 65)

for name, module in packages.items():
    try:
        importlib.import_module(module)

        package_name = {
            "scikit-learn": "scikit-learn",
            "XGBoost": "xgboost",
            "SHAP": "shap",
            "NumPy": "numpy",
            "GeoPandas": "geopandas",
            "Rasterio": "rasterio",
            "Shapely": "shapely",
            "Fiona": "fiona",
            "GDAL": "gdal",
            "Numba": "numba",
            "psycopg2-binary": "psycopg2-binary",
            "SQLAlchemy": "SQLAlchemy",
            "GeoAlchemy2": "GeoAlchemy2",
            "Jupyter": "jupyter",
            "ipykernel": "ipykernel"
        }.get(name, module)

        try:
            ver = version(package_name)
        except PackageNotFoundError:
            ver = "installed"

        print(f"{name:<20} OK ({ver})")

    except Exception as e:
        print(f"{name:<20} FAILED")
        print(f"   Error: {e}")
        all_ok = False

print("\n" + "=" * 65)

if all_ok:
    print("All libraries OK.")
else:
    print("Some libraries require attention.")

print("=" * 65)