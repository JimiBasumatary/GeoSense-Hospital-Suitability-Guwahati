import geopandas as gpd
import pandas as pd
from pathlib import Path

from image_classifier import classify_imagery


# ============================================================
# PHASE 1 → PHASE 2 INTEGRATION
# ============================================================

INPUT = Path("data/outputs/geosense_ranked_candidates.gpkg")
OUTPUT = Path("data/outputs/geosense_phase1_phase2_integrated.gpkg")


# ============================================================
# LOAD PHASE 1 CANDIDATES
# ============================================================

gdf = gpd.read_file(INPUT)

print("=" * 70)
print("PHASE 1 + PHASE 2 INTEGRATION")
print("=" * 70)

print(f"Phase 1 candidates: {len(gdf)}")
print(f"Original CRS: {gdf.crs}")


# ============================================================
# CALCULATE CENTROIDS CORRECTLY
# ============================================================

# Reproject before calculating centroids
projected = gdf.to_crs("EPSG:32646")

centroids = projected.geometry.centroid

# Convert centroids back to WGS84
centroids_wgs84 = gpd.GeoSeries(
    centroids,
    crs="EPSG:32646"
).to_crs("EPSG:4326")


# ============================================================
# EXTRACT PHASE 2 SATELLITE INFORMATION
# ============================================================

results = []

for i, point in enumerate(centroids_wgs84):

    lat = point.y
    lon = point.x

    print(
        f"\n[{i + 1}/{len(gdf)}] "
        f"OSM ID: {gdf.iloc[i]['osm_id']} "
        f"Lat: {lat:.6f} "
        f"Lon: {lon:.6f}"
    )

    try:

        result = classify_imagery(
            lat=lat,
            lon=lon,
            radius_m=500
        )

        results.append({
            "phase2_land_cover": result.get("land_cover"),
            "phase2_class_id": result.get("class_id"),
            "phase2_confidence": result.get("confidence_pct"),
            "phase2_ndvi": result.get("ndvi"),
            "phase2_ndwi": result.get("ndwi"),
            "phase2_ndvi_label": result.get("ndvi_label"),
            "phase2_ndvi_2015": result.get("ndvi_2015"),
            "phase2_ndvi_2023": result.get("ndvi_2023"),
            "phase2_ndvi_difference": result.get("ndvi_difference"),
            "phase2_change_flag": result.get("change_flag"),
            "phase2_change_description": result.get(
                "change_description"
            ),
            "phase2_model": result.get("model"),
            "phase2_inference_seconds": result.get(
                "inference_time_seconds"
            )
        })

    except Exception as e:

        print(f"ERROR: {e}")

        results.append({
            "phase2_land_cover": None,
            "phase2_class_id": None,
            "phase2_confidence": None,
            "phase2_ndvi": None,
            "phase2_ndwi": None,
            "phase2_ndvi_label": None,
            "phase2_ndvi_2015": None,
            "phase2_ndvi_2023": None,
            "phase2_ndvi_difference": None,
            "phase2_change_flag": None,
            "phase2_change_description": str(e),
            "phase2_model": None,
            "phase2_inference_seconds": None
        })


# ============================================================
# COMBINE PHASE 1 + PHASE 2
# ============================================================

phase2_df = pd.DataFrame(results)

integrated = pd.concat(
    [
        gdf.reset_index(drop=True),
        phase2_df
    ],
    axis=1
)


# ============================================================
# SAVE NEW INTEGRATED DATASET
# ============================================================

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

integrated.to_file(
    OUTPUT,
    driver="GPKG"
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("INTEGRATION COMPLETE")
print("=" * 70)

print(f"Output: {OUTPUT}")
print(f"Records: {len(integrated)}")

print("\nNew Phase 2 fields:")

for column in phase2_df.columns:
    print(f"  - {column}")

print("\nPhase 2 land-cover distribution:")

print(
    integrated["phase2_land_cover"]
    .value_counts(dropna=False)
)

print("\nChange detection:")

print(
    integrated["phase2_change_flag"]
    .value_counts(dropna=False)
)

print("\nOriginal Phase 1 file was NOT modified.")