import os
import shutil
import zipfile
import rasterio


# ============================================================
# GeoSense Phase 2
# Prepare Compact Colab Dataset
# Study Area: Guwahati
# Year: 2023
# ============================================================

SOURCE_CHIPS = r"data\satellite\chips\2023"
SOURCE_LABEL = r"data\satellite\labels\labels_2023.npy"

OUTPUT_DIR = r"data\satellite\colab_dataset"
OUTPUT_CHIPS = os.path.join(OUTPUT_DIR, "chips", "2023")
OUTPUT_LABELS = os.path.join(OUTPUT_DIR, "labels")

ZIP_FILE = r"data\satellite\guwahati_unet_colab_2023.zip"


os.makedirs(OUTPUT_CHIPS, exist_ok=True)
os.makedirs(OUTPUT_LABELS, exist_ok=True)


print("=" * 70)
print("GeoSense Phase 2 - Prepare Colab Dataset")
print("=" * 70)

print()
print("Source chips:", SOURCE_CHIPS)
print("Source label:", SOURCE_LABEL)
print()


# ------------------------------------------------------------
# Copy only the first 6 bands from each TIFF
# ------------------------------------------------------------

chip_files = sorted(
    [
        f for f in os.listdir(SOURCE_CHIPS)
        if f.lower().endswith(".tif")
    ]
)

print("Number of source chips:", len(chip_files))
print()

for i, filename in enumerate(chip_files, start=1):

    source_file = os.path.join(
        SOURCE_CHIPS,
        filename
    )

    output_file = os.path.join(
        OUTPUT_CHIPS,
        filename
    )

    with rasterio.open(source_file) as src:

        profile = src.profile.copy()

        profile.update(
            count=6,
            dtype="float32",
            compress="deflate"
        )

        with rasterio.open(
            output_file,
            "w",
            **profile
        ) as dst:

            data = src.read(
                indexes=[1, 2, 3, 4, 5, 6]
            ).astype("float32")

            dst.write(data)

    if i % 50 == 0 or i == len(chip_files):
        print(
            f"Processed {i}/{len(chip_files)} chips"
        )


# ------------------------------------------------------------
# Copy label file
# ------------------------------------------------------------

print()
print("Copying label file...")

shutil.copy2(
    SOURCE_LABEL,
    os.path.join(
        OUTPUT_LABELS,
        "labels_2023.npy"
    )
)

print("Label copied successfully.")


# ------------------------------------------------------------
# Create ZIP
# ------------------------------------------------------------

print()
print("Creating compact ZIP...")
print("This may take a few minutes.")


if os.path.exists(ZIP_FILE):
    os.remove(ZIP_FILE)


with zipfile.ZipFile(
    ZIP_FILE,
    "w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=6
) as zipf:

    for root, dirs, files in os.walk(
        OUTPUT_DIR
    ):

        for filename in files:

            full_path = os.path.join(
                root,
                filename
            )

            archive_name = os.path.relpath(
                full_path,
                OUTPUT_DIR
            )

            zipf.write(
                full_path,
                archive_name
            )


# ------------------------------------------------------------
# Final information
# ------------------------------------------------------------

zip_size_mb = (
    os.path.getsize(ZIP_FILE)
    / (1024 * 1024)
)

print()
print("=" * 70)
print("COMPACT DATASET CREATED")
print("=" * 70)

print()
print("Output directory:")
print(OUTPUT_DIR)

print()
print("ZIP file:")
print(ZIP_FILE)

print()
print(
    f"ZIP size: {zip_size_mb:.2f} MB"
)

print()
print("Training chips:", len(chip_files))
print("Input bands per chip: 6")
print("Label file: labels_2023.npy")

print()
print("=" * 70)
print("READY FOR COLAB")
print("=" * 70)