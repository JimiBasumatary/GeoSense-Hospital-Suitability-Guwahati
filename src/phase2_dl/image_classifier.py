import os
import time
import json

import numpy as np
import rasterio
import torch
import segmentation_models_pytorch as smp


# ============================================================
# GeoSense Agent 2.0
# Phase 2 - Exercise 5.1
# Image Classifier + Change Detection
# Study Area: Guwahati
# ============================================================


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

IMAGE_2015 = (
    r"data\satellite\processed"
    r"\sentinel2_guwahati_2015_processed.tif"
)

IMAGE_2023 = (
    r"data\satellite\processed"
    r"\sentinel2_guwahati_2023_processed.tif"
)

PRITHVI_MODEL = r"models\saved\prithvi_finetuned.pth"
UNET_MODEL = r"models\saved\unet_best.pth"


# ------------------------------------------------------------
# Model configuration
# ------------------------------------------------------------

NUM_CLASSES = 5
IN_CHANNELS = 6
CHIP_SIZE = 224

CLASS_NAMES = [
    "Urban",
    "Vegetation",
    "Water",
    "Bare Land",
    "Agriculture"
]


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ------------------------------------------------------------
# Load trained model
# ------------------------------------------------------------

def _load_model():

    print("=" * 70)
    print("Loading GeoSense segmentation model")
    print("=" * 70)

    # Preferred model:
    # documented foundation-model experiment
    if os.path.exists(PRITHVI_MODEL):

        print("Loading foundation-model experiment:")
        print(PRITHVI_MODEL)

        model = smp.Unet(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=IN_CHANNELS,
            classes=NUM_CLASSES
        )

        checkpoint = torch.load(
            PRITHVI_MODEL,
            map_location=DEVICE
        )

        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)

        selected_model = "foundation_resnet50_substitute"

    elif os.path.exists(UNET_MODEL):

        print("Foundation model file not found.")
        print("Falling back to U-Net:")
        print(UNET_MODEL)

        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=IN_CHANNELS,
            classes=NUM_CLASSES
        )

        checkpoint = torch.load(
            UNET_MODEL,
            map_location=DEVICE
        )

        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)

        selected_model = "unet_resnet34"

    else:
        raise FileNotFoundError(
            "Neither prithvi_finetuned.pth nor "
            "unet_best.pth was found."
        )

    model = model.to(DEVICE)
    model.eval()

    print("Selected model:", selected_model)
    print("Device:", DEVICE)

    return model, selected_model


# ------------------------------------------------------------
# Convert latitude/longitude to image patch
# ------------------------------------------------------------

def _get_image_patch(lat, lon, radius_m):

    with rasterio.open(IMAGE_2023) as src:

        # Convert radius from metres to approximate degrees.
        # This is appropriate for the EPSG:4326 raster at
        # the Guwahati study-area scale.

        meters_per_degree_lat = 111320.0

        meters_per_degree_lon = (
            111320.0 * np.cos(np.radians(lat))
        )

        lat_radius = radius_m / meters_per_degree_lat
        lon_radius = radius_m / meters_per_degree_lon

        left = lon - lon_radius
        right = lon + lon_radius
        bottom = lat - lat_radius
        top = lat + lat_radius

        # Convert geographic coordinates to pixel coordinates

        row_top, col_left = src.index(left, top)
        row_bottom, col_right = src.index(right, bottom)

        center_row, center_col = src.index(lon, lat)

        half_size = CHIP_SIZE // 2

        row_start = center_row - half_size
        col_start = center_col - half_size

        row_end = row_start + CHIP_SIZE
        col_end = col_start + CHIP_SIZE

        # Keep the window inside the raster

        row_start = max(0, row_start)
        col_start = max(0, col_start)

        row_end = min(src.height, row_end)
        col_end = min(src.width, col_end)

        window_height = row_end - row_start
        window_width = col_end - col_start

        data = src.read(
            indexes=[1, 2, 3, 4, 5, 6],
            window=rasterio.windows.Window(
                col_start,
                row_start,
                window_width,
                window_height
            )
        ).astype(np.float32)

    # Create exactly 224 x 224 x 6

    patch = np.full(
        (6, CHIP_SIZE, CHIP_SIZE),
        np.nan,
        dtype=np.float32
    )

    copy_height = min(CHIP_SIZE, data.shape[1])
    copy_width = min(CHIP_SIZE, data.shape[2])

    patch[
        :,
        :copy_height,
        :copy_width
    ] = data[
        :,
        :copy_height,
        :copy_width
    ]

    # Fill NaNs with the mean of each band

    for band in range(6):

        band_data = patch[band]

        valid = np.isfinite(band_data)

        if np.any(valid):
            band_mean = np.mean(
                band_data[valid]
            )
        else:
            band_mean = 0.0

        band_data[~valid] = band_mean

        patch[band] = band_data

    patch = np.clip(
        patch,
        0,
        1
    )

    return patch


# ------------------------------------------------------------
# Compute NDVI and NDWI
# ------------------------------------------------------------

def _compute_ndvi_ndwi(patch):

    # Band positions inside the six-band patch:
    #
    # 0 = B2 Blue
    # 1 = B3 Green
    # 2 = B4 Red
    # 3 = B8 NIR
    # 4 = B11 SWIR1
    # 5 = B12 SWIR2

    green = patch[1]
    red = patch[2]
    nir = patch[3]

    # NDVI

    ndvi_denominator = nir + red

    ndvi = np.divide(
        nir - red,
        ndvi_denominator,
        out=np.zeros_like(nir),
        where=ndvi_denominator != 0
    )

    # NDWI

    ndwi_denominator = green + nir

    ndwi = np.divide(
        green - nir,
        ndwi_denominator,
        out=np.zeros_like(green),
        where=ndwi_denominator != 0
    )

    return ndvi, ndwi


# ------------------------------------------------------------
# Read corresponding 2015 patch
# ------------------------------------------------------------

def _get_patch_from_raster(
    raster_path,
    lat,
    lon
):

    with rasterio.open(raster_path) as src:

        center_row, center_col = src.index(
            lon,
            lat
        )

        half_size = CHIP_SIZE // 2

        row_start = center_row - half_size
        col_start = center_col - half_size

        row_end = row_start + CHIP_SIZE
        col_end = col_start + CHIP_SIZE

        # Shift window if close to raster edge

        if row_start < 0:
            row_start = 0
            row_end = CHIP_SIZE

        if col_start < 0:
            col_start = 0
            col_end = CHIP_SIZE

        if row_end > src.height:
            row_end = src.height
            row_start = max(
                0,
                row_end - CHIP_SIZE
            )

        if col_end > src.width:
            col_end = src.width
            col_start = max(
                0,
                col_end - CHIP_SIZE
            )

        data = src.read(
            indexes=[1, 2, 3, 4, 5, 6],
            window=rasterio.windows.Window(
                col_start,
                row_start,
                CHIP_SIZE,
                CHIP_SIZE
            )
        ).astype(np.float32)

    # Fill NaNs

    for band in range(6):

        valid = np.isfinite(data[band])

        if np.any(valid):
            mean_value = np.mean(
                data[band][valid]
            )
        else:
            mean_value = 0.0

        data[band][~valid] = mean_value

    data = np.clip(data, 0, 1)

    return data


# ------------------------------------------------------------
# Classify imagery
# ------------------------------------------------------------

def classify_imagery(
    lat,
    lon,
    radius_m=500
):

    start_time = time.time()

    print()
    print("=" * 70)
    print("GeoSense Image Classification")
    print("=" * 70)

    print("Latitude:", lat)
    print("Longitude:", lon)
    print("Radius:", radius_m, "m")

    # Load model

    model, model_name = _load_model()

    # Get 2023 image patch

    patch_2023 = _get_image_patch(
        lat,
        lon,
        radius_m
    )

    # Compute indices

    ndvi_2023, ndwi_2023 = (
        _compute_ndvi_ndwi(
            patch_2023
        )
    )

    # Convert patch to tensor

    input_tensor = torch.from_numpy(
        patch_2023
    ).unsqueeze(0).float()

    input_tensor = input_tensor.to(DEVICE)

    # Model inference

    with torch.no_grad():

        output = model(
            input_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        prediction = torch.argmax(
            probabilities,
            dim=1
        )[0]

    prediction_array = (
        prediction.cpu().numpy()
    )

    probability_array = (
        probabilities[0]
        .cpu()
        .numpy()
    )

    # Class distribution

    total_pixels = prediction_array.size

    class_distribution = {}

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        count = np.sum(
            prediction_array == class_id
        )

        percentage = (
            count / total_pixels
        ) * 100

        class_distribution[class_name] = round(
            float(percentage),
            2
        )

    # Dominant class

    dominant_class_id = int(
        np.bincount(
            prediction_array.flatten(),
            minlength=NUM_CLASSES
        ).argmax()
    )

    dominant_class = CLASS_NAMES[
        dominant_class_id
    ]

    # Confidence

    confidence = float(
        probability_array[
            dominant_class_id
        ].mean()
    )

    confidence_pct = confidence * 100

    # Mean NDVI / NDWI

    mean_ndvi_2023 = float(
        np.nanmean(ndvi_2023)
    )

    mean_ndwi_2023 = float(
        np.nanmean(ndwi_2023)
    )

    # Simple NDVI interpretation

    if mean_ndvi_2023 < 0:
        ndvi_label = "Water / Non-vegetated surface"
    elif mean_ndvi_2023 < 0.15:
        ndvi_label = "Bare / Sparse vegetation"
    elif mean_ndvi_2023 < 0.4:
        ndvi_label = "Moderate vegetation / Agriculture"
    else:
        ndvi_label = "Dense vegetation"

    # --------------------------------------------------------
    # Change detection
    # --------------------------------------------------------

    patch_2015 = _get_patch_from_raster(
        IMAGE_2015,
        lat,
        lon
    )

    ndvi_2015, ndwi_2015 = (
        _compute_ndvi_ndwi(
            patch_2015
        )
    )

    mean_ndvi_2015 = float(
        np.nanmean(ndvi_2015)
    )

    ndvi_difference = abs(
        mean_ndvi_2023 -
        mean_ndvi_2015
    )

    change_flag = (
        ndvi_difference > 0.15
    )

    if change_flag:

        change_description = (
            "Significant NDVI change detected "
            f"between 2015 and 2023 "
            f"(absolute difference = "
            f"{ndvi_difference:.3f})."
        )

    else:

        change_description = (
            "No significant NDVI change detected "
            f"between 2015 and 2023 "
            f"(absolute difference = "
            f"{ndvi_difference:.3f})."
        )

    elapsed_time = (
        time.time() - start_time
    )

    # --------------------------------------------------------
    # Final structured result
    # --------------------------------------------------------

    result = {

        "land_cover": dominant_class,

        "class_id": dominant_class_id,

        "confidence_pct": round(
            confidence_pct,
            2
        ),

        "ndvi": round(
            mean_ndvi_2023,
            4
        ),

        "ndwi": round(
            mean_ndwi_2023,
            4
        ),

        "ndvi_label": ndvi_label,

        "class_distribution":
            class_distribution,

        "change_flag":
            bool(change_flag),

        "change_description":
            change_description,

        "ndvi_2015":
            round(
                mean_ndvi_2015,
                4
            ),

        "ndvi_2023":
            round(
                mean_ndvi_2023,
                4
            ),

        "ndvi_difference":
            round(
                ndvi_difference,
                4
            ),

        "model":
            model_name,

        "inference_time_seconds":
            round(
                elapsed_time,
                3
            )
    }

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CLASSIFICATION RESULT")
    print("=" * 70)

    print(
        json.dumps(
            result,
            indent=4
        )
    )

    print()
    print(
        "Inference time:",
        round(elapsed_time, 3),
        "seconds"
    )

    print("=" * 70)

    return result


# ------------------------------------------------------------
# Test
# ------------------------------------------------------------

if __name__ == "__main__":

    # Guwahati test coordinate
    # Located inside the project study-area bounds.

    result = classify_imagery(
    26.2314800,
    91.8962612,
    radius_m=500
)