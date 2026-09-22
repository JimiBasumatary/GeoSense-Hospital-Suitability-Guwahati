import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import os

# ---------------------------------------------------------
# INPUTS
# ---------------------------------------------------------
CANDIDATES = r"data\outputs\geosense_phase1_phase2_integrated.gpkg"
BOUNDARY = r"data\boundaries\guwahati\guwahati.shp"
OUTPUT = r"outputs\plots\final_geosense_integrated_map.png"

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

# ---------------------------------------------------------
# 1. READ DATA
# ---------------------------------------------------------
print("=" * 70)
print("GeoSense - Final Integrated Hospital Suitability Map")
print("=" * 70)

candidates = gpd.read_file(CANDIDATES)
boundary = gpd.read_file(BOUNDARY)

print("Candidates:", len(candidates))
print("Candidate CRS:", candidates.crs)
print("Boundary CRS:", boundary.crs)

# ---------------------------------------------------------
# 2. REPROJECT TO PROJECTED CRS FOR MAPPING
# ---------------------------------------------------------
target_crs = "EPSG:32646"

candidates = candidates.to_crs(target_crs)
boundary = boundary.to_crs(target_crs)

# ---------------------------------------------------------
# 3. CREATE FIGURE
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 10))

# Boundary
boundary.plot(
    ax=ax,
    facecolor="none",
    edgecolor="black",
    linewidth=1.5
)

# Candidate locations
candidates.plot(
    ax=ax,
    column="suitability_score",
    cmap="viridis",
    legend=True,
    markersize=70,
    edgecolor="black",
    linewidth=0.5
)

# ---------------------------------------------------------
# 4. LABEL CANDIDATE RANK
# ---------------------------------------------------------
for _, row in candidates.iterrows():

    point = row.geometry.centroid

    ax.annotate(
        str(int(row["rank"])),
        (point.x, point.y),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=7,
        fontweight="bold"
    )

# ---------------------------------------------------------
# 5. NORTH ARROW
# ---------------------------------------------------------
ax.annotate(
    "N",
    xy=(0.95, 0.92),
    xycoords="axes fraction",
    ha="center",
    va="center",
    fontsize=20,
    fontweight="bold"
)

ax.arrow(
    0.95,
    0.84,
    0,
    0.06,
    transform=ax.transAxes,
    width=0.003,
    head_width=0.025,
    head_length=0.025,
    length_includes_head=True
)

# ---------------------------------------------------------
# 6. SCALE BAR
# ---------------------------------------------------------
try:
    scalebar = ScaleBar(
        1,
        units="m",
        location="lower left",
        box_alpha=0.8
    )
    ax.add_artist(scalebar)
except Exception:
    print("Scale bar could not be added.")

# ---------------------------------------------------------
# 7. TITLE
# ---------------------------------------------------------
ax.set_title(
    "Integrated Hospital Site Suitability Map – Guwahati",
    fontsize=16,
    fontweight="bold",
    pad=15
)

ax.text(
    0.5,
    1.01,
    "Phase 1 GIS/PostGIS + Random Forest and Phase 2 Sentinel-2 + Deep Learning",
    transform=ax.transAxes,
    ha="center",
    fontsize=10
)

# ---------------------------------------------------------
# 8. AXIS LABELS
# ---------------------------------------------------------
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")

# ---------------------------------------------------------
# 9. INFORMATION BOX
# ---------------------------------------------------------
info = (
    "GeoSense Hospital Candidates\n"
    f"Total candidates: {len(candidates)}\n"
    "Symbol colour: Phase 1 suitability score\n"
    "Labels: Candidate rank\n"
    "Phase 2: NDVI, NDWI and change detection\n"
    "CRS: EPSG:32646"
)

ax.text(
    0.02,
    0.02,
    info,
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment="bottom",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.85
    )
)

# ---------------------------------------------------------
# 10. GRID
# ---------------------------------------------------------
ax.grid(
    True,
    linestyle="--",
    alpha=0.3
)

# ---------------------------------------------------------
# 11. SAVE
# ---------------------------------------------------------
plt.tight_layout()

plt.savefig(
    OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\n" + "=" * 70)
print("FINAL MAP CREATED SUCCESSFULLY")
print("=" * 70)
print("Candidates mapped:", len(candidates))
print("Output:", OUTPUT)
print("CRS:", target_crs)
print("=" * 70)