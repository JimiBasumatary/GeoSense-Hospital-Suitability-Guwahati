import os
from pathlib import Path
# GeoSense Phase 1 - Exercise 3
# Data Inventory Verification
print("=" * 60)
print("GEOSENSE PHASE 1 - EXERCISE 3")
print("DATA INVENTORY VERIFICATION")
print("=" * 60)
# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Expected directories
directories = [
    "data",
    "docs",
    "models",
    "notebooks",
    "outputs",
    "src",
]
print("\n1. PROJECT DIRECTORY CHECK")
print("-" * 60)
for directory in directories:
    path = PROJECT_ROOT / directory
    if path.exists():
        print(f"[OK] {directory}")
    else:
        print(f"[MISSING] {directory}")
# ==========================================
# Data directory inspection
# ==========================================
print("\n2. DATA DIRECTORY INVENTORY")
print("-" * 60)
data_path = PROJECT_ROOT / "data"
if data_path.exists():
    for root, dirs, files in os.walk(data_path):
        relative_path = Path(root).relative_to(PROJECT_ROOT)
        print(f"\nFolder: {relative_path}")
        if files:
            for file in files:
                print(f"   - {file}")
        else:
            print("   (No files found)")
else:
    print("[MISSING] Data directory not found")
 # Important file types
print("\n3. SPATIAL DATA FILE SUMMARY")
print("-" * 60)
extensions = {
    ".tif": "GeoTIFF / Raster",
    ".tiff": "GeoTIFF / Raster",
    ".gpkg": "GeoPackage",
    ".geojson": "GeoJSON",
    ".shp": "Shapefile",
    ".csv": "CSV",
}
file_counts = {}
for extension in extensions:
    file_counts[extension] = []
for root, dirs, files in os.walk(PROJECT_ROOT / "data"):
    for file in files:
        file_path = Path(root) / file
        extension = file_path.suffix.lower()
        if extension in file_counts:
            file_counts[extension].append(file_path)
for extension, files in file_counts.items():
    dataset_type = extensions[extension]
    print(f"\n{dataset_type} ({extension})")
    if files:
        for file in files:
            relative_file = file.relative_to(PROJECT_ROOT)
            print(f"   [FOUND] {relative_file}")
    else:
        print("   No files found")
# Final status
print("\n" + "=" * 60)
print("DATA INVENTORY CHECK COMPLETED")
print("=" * 60)