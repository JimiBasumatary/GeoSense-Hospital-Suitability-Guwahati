
import os
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.mask import mask

BASE = r"D:\GeoSense_Agent"

BOUNDARY = os.path.join(
    BASE, "data", "boundaries", "guwahati", "guwahati.shp"
)

RASTER_DIR = os.path.join(
    BASE, "data", "satellite", "processed"
)

OUTPUT_DIR = os.path.join(RASTER_DIR, "clipped")
os.makedirs(OUTPUT_DIR, exist_ok=True)

years = ["2015", "2023"]

# Read the Guwahati study-area boundary
boundary = gpd.read_file(BOUNDARY)

if boundary.empty:
    raise ValueError("The Guwahati boundary shapefile is empty.")

if boundary.crs is None:
    raise ValueError("The boundary CRS is undefined.")

# Repair invalid geometry if needed
boundary["geometry"] = boundary.geometry.make_valid()
boundary = boundary[
    boundary.geometry.notna() & ~boundary.geometry.is_empty
]

if boundary.empty:
    raise ValueError("No valid boundary geometry remains.")

for year in years:
    input_path = os.path.join(
        RASTER_DIR, f"ndvi_guwahati_{year}.tif"
    )
    output_path = os.path.join(
        OUTPUT_DIR, f"ndvi_guwahati_{year}_clipped.tif"
    )

    print(f"\nClipping NDVI for {year}...")

    with rasterio.open(input_path) as src:
        # Reproject the boundary to match the raster CRS
        boundary_raster_crs = boundary.to_crs(src.crs)

        geometries = [
            geom.__geo_interface__
            for geom in boundary_raster_crs.geometry
        ]

        clipped, transform = mask(
            src,
            geometries,
            crop=True,
            filled=True,
            nodata=-9999.0
        )

        profile = src.profile.copy()
        profile.update(
            height=clipped.shape[1],
            width=clipped.shape[2],
            transform=transform,
            count=1,
            dtype="float32",
            nodata=-9999.0,
            compress="deflate"
        )

        # Ensure pixels outside the boundary are NoData
        values = clipped[0].astype("float32")
        values[~np.isfinite(values)] = -9999.0

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(values, 1)
            dst.set_band_description(1, f"NDVI_{year}_clipped")

    valid = values != -9999.0

    print("Saved:", output_path)
    print("Clipped dimensions:", values.shape[1], "x", values.shape[0])
    print("Valid pixels inside clip extent:", int(valid.sum()))

print("\nClipping completed for both years.")
