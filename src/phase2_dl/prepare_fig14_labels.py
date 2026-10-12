
import os
import numpy as np
import rasterio

root = r"D:\GeoSense_Agent"

image_file = os.path.join(
    root, "data", "satellite", "processed",
    "sentinel2_guwahati_2023_processed.tif"
)
label_file = os.path.join(
    root, "data", "satellite", "labels", "labels_2023.npy"
)
output_file = os.path.join(
    root, "outputs", "guwahati_2023_rule_labels_fig14.tif"
)

labels = np.load(label_file)

with rasterio.open(image_file) as src:
    if labels.shape != (src.height, src.width):
        raise ValueError(
            f"Label dimensions {labels.shape} do not match "
            f"raster dimensions {(src.height, src.width)}"
        )

    profile = src.profile.copy()

profile.update(
    driver="GTiff",
    count=1,
    dtype="uint8",
    nodata=255,
    compress="deflate"
)

with rasterio.open(output_file, "w", **profile) as dst:
    dst.write(labels.astype(np.uint8), 1)

print("Rule-label GeoTIFF created:", output_file)
print("Dimensions:", labels.shape)
