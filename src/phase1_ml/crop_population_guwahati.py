import rasterio
from rasterio.mask import mask
import geopandas as gpd
from pathlib import Path


INPUT_RASTER = (
    r"data\population\raw\population"
    r"\ind_pop_2024_UC_100m_R2024A_v1.tif"
)

BOUNDARY = r"data\boundaries\guwahati\guwahati.shp"

OUTPUT_RASTER = (
    r"data\population\processed"
    r"\population_guwahati_raw_2024.tif"
)


def crop_population():

    print("\n" + "=" * 60)
    print("GeoSense - Crop Population Raster to Guwahati")
    print("=" * 60)

    print("\nReading Guwahati boundary...")

    boundary = gpd.read_file(BOUNDARY)

    print("Boundary CRS:", boundary.crs)

    print("\nOpening population raster...")

    with rasterio.open(INPUT_RASTER) as src:

        print("Population CRS:", src.crs)

        # Convert boundary to raster CRS
        boundary = boundary.to_crs(src.crs)

        print("\nCropping raster...")

        geometries = boundary.geometry.values

        cropped, transform = mask(
            src,
            geometries,
            crop=True,
            nodata=src.nodata
        )

        profile = src.profile.copy()

        profile.update({
            "height": cropped.shape[1],
            "width": cropped.shape[2],
            "transform": transform,
            "compress": "lzw"
        })

        Path(OUTPUT_RASTER).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with rasterio.open(
            OUTPUT_RASTER,
            "w",
            **profile
        ) as dst:

            dst.write(cropped)

    print("\nSUCCESS!")
    print("Guwahati population raster created:")
    print(OUTPUT_RASTER)


if __name__ == "__main__":
    crop_population()