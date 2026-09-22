import geopandas as gpd
import matplotlib.pyplot as plt

INPUT = r"data\outputs\geosense_phase1_phase2_integrated.gpkg"
OUTPUT = r"outputs\plots\final_integrated_hospital_suitability_map.png"

print("=" * 70)
print("GeoSense - Final Phase 1 + Phase 2 Integrated Map")
print("=" * 70)

# ---------------------------------------------------------
# 1. Read integrated candidate layer
# ---------------------------------------------------------
gdf = gpd.read_file(INPUT)

print("Candidates:", len(gdf))
print("CRS:", gdf.crs)

# ---------------------------------------------------------
# 2. Create map
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 10))

gdf.plot(
    ax=ax,
    column="suitability_score",
    cmap="viridis",
    legend=True,
    markersize=70,
    edgecolor="black"
)

# ---------------------------------------------------------
# 3. Add candidate rank labels
# ---------------------------------------------------------
for _, row in gdf.iterrows():
    x = row.geometry.centroid.x
    y = row.geometry.centroid.y

    ax.text(
        x,
        y,
        str(int(row["rank"])),
        fontsize=7,
        ha="center",
        va="center"
    )

# ---------------------------------------------------------
# 4. Map formatting
# ---------------------------------------------------------
ax.set_title(
    "Integrated Hospital Site Suitability Map - Guwahati\n"
    "Phase 1 GIS/PostGIS + Phase 2 Satellite Analysis",
    fontsize=14
)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

ax.grid(True, alpha=0.3)

plt.tight_layout()

# ---------------------------------------------------------
# 5. Save
# ---------------------------------------------------------
plt.savefig(
    OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nMAP CREATED SUCCESSFULLY")
print("Output:", OUTPUT)
print("Candidates mapped:", len(gdf))
print("=" * 70)