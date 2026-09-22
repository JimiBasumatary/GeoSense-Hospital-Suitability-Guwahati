import os
import shutil
import zipfile
import rasterio


# ============================================================
# GeoSense Phase 2
# Small Colab Training Dataset
# Study Area: Guwahati
# Year: 2023
# ============================================================

SOURCE_CHIPS = r"data\satellite\chips\2023"
SOURCE_LABEL = r"data\satellite\labels\labels_2023.npy"

OUTPUT_DIR = r"data\satellite\colab_small"
OUTPUT_CHIPS = os.path.join(OUTPUT_DIR, "chips", "2023")
OUTPUT_LABELS = os.path.join(OUTPUT_DIR, "labels")

ZIP_FILE = r"data\satellite\guwahati_unet_colab_small.zip"

NUMBER_OF_CHIPS = 80


os.makedirs(OUTPUT_CHIPS, exist_ok=True)
os.makedirs(OUTPUT_LABELS, exist_ok=True)


print("=" * 70)
print("GeoSense Phase 2 - Small Colab Dataset")
print("=" * 70)

# ------------------------------------------------------------
# Find all TIFF chips
# ------------------------------------------------------------

all_chips = sorted(
    [
        f for f in os.listdir(SOURCE_CHIPS)
        if f.lower().endswith(".tif")
    ]
)

print()
print("Total available chips:", len(all_chips))


# ------------------------------------------------------------
# Select evenly distributed chips
# ------------------------------------------------------------

step = max(1, len(all_chips) // NUMBER_OF_CHIPS)

selected_chips = all_chips[::step]

# Make sure exactly NUMBER_OF_CHIPS are selected
selected_chips = selected_chips[:NUMBER_OF_CHIPS]

print("Selected chips:", len(selected_chips))


# ------------------------------------------------------------
# Copy selected chips
# Keep only 6 spectral bands
# ------------------------------------------------------------

print()
print("Preparing selected chips...")

for i, filename in enumerate(selected_chips, start=1):

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

    if i % 10 == 0 or i == len(selected_chips):
        print(
            f"Prepared {i}/{len(selected_chips)} chips"
        )


# ------------------------------------------------------------
# Copy labels
# ------------------------------------------------------------

print()
print("Copying labels...")

shutil.copy2(
    SOURCE_LABEL,
    os.path.join(
        OUTPUT_LABELS,
        "labels_2023.npy"
    )
)

print("Labels copied.")


# ------------------------------------------------------------
# Create ZIP
# ------------------------------------------------------------

print()
print("Creating ZIP...")
print("Please wait...")

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
print("SMALL COLAB DATASET CREATED")
print("=" * 70)

print()
print("Selected chips:", len(selected_chips))
print("Input bands per chip: 6")
print("Label file: labels_2023.npy")

print()
print("ZIP file:")
print(ZIP_FILE)

print()
print(
    f"ZIP size: {zip_size_mb:.2f} MB"
)

print()
print("=" * 70)
print("READY FOR COLAB")
print("=" * 70)