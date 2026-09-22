import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp


# ============================================================
# GeoSense Phase 2 - Exercise 4.2
# U-Net vs Foundation Model Evaluation
# Study Area: Guwahati
# ============================================================

CHIP_DIR = r"data\satellite\colab_small\chips\2023"
LABEL_FILE = r"data\satellite\labels\labels_2023.npy"

UNET_MODEL = r"models\saved\unet_best.pth"

FOUNDATION_MODEL = (
    r"models\saved\prithvi_finetuned.pth"
)

OUTPUT_DIR = r"outputs\plots"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# Configuration
# ============================================================

NUM_CLASSES = 5
IN_CHANNELS = 6
CHIP_SIZE = 224
BATCH_SIZE = 8

CLASS_NAMES = [
    "Urban",
    "Vegetation",
    "Water",
    "Bare Land",
    "Agriculture"
]


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("GeoSense Phase 2 - Exercise 4.2")
print("U-Net vs Foundation Model Evaluation")
print("Study Area: Guwahati")
print("=" * 70)

print()
print("Device:", device)


# ============================================================
# Dataset
# ============================================================

class SatelliteDataset(Dataset):

    def __init__(
        self,
        chip_dir,
        label_file,
        indices
    ):

        self.chip_dir = chip_dir

        self.label_array = np.load(
            label_file,
            mmap_mode="r"
        )

        all_chips = sorted(
            [
                f
                for f in os.listdir(chip_dir)
                if f.lower().endswith(".tif")
            ]
        )

        self.chips = [
            all_chips[i]
            for i in indices
        ]


    def __len__(self):

        return len(self.chips)


    def __getitem__(self, index):

        import rasterio

        chip_name = self.chips[index]

        chip_path = os.path.join(
            self.chip_dir,
            chip_name
        )

        with rasterio.open(chip_path) as src:

            image = src.read(
                indexes=[1, 2, 3, 4, 5, 6]
            ).astype(np.float32)


        # Filename example:
        # guwahati_2023_row_00000_col_00000.tif

        filename = chip_name.replace(
            ".tif",
            ""
        )

        parts = filename.split("_")

        row = int(parts[3])
        col = int(parts[5])


        label = self.label_array[
            row:row + CHIP_SIZE,
            col:col + CHIP_SIZE
        ]


        # ----------------------------------------------------
        # Handle invalid pixels
        # ----------------------------------------------------

        for band in range(
            image.shape[0]
        ):

            band_data = image[band]

            valid = np.isfinite(
                band_data
            )

            if valid.any():

                mean_value = np.nanmean(
                    band_data
                )

                band_data[
                    ~valid
                ] = mean_value

            else:

                band_data[:] = 0


        image = np.nan_to_num(
            image,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )

        image = np.clip(
            image,
            0,
            1
        )


        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        label = torch.tensor(
            label.copy(),
            dtype=torch.long
        )


        return image, label


# ============================================================
# Prepare validation/test set
# ============================================================

all_chips = sorted(
    [
        f
        for f in os.listdir(CHIP_DIR)
        if f.lower().endswith(".tif")
    ]
)


rng = np.random.default_rng(42)

indices = np.arange(
    len(all_chips)
)

rng.shuffle(indices)

split = int(
    0.8 * len(indices)
)

train_indices = indices[:split]

test_indices = indices[split:]


print()
print("Total chips:", len(all_chips))
print("Training chips:", len(train_indices))
print("Held-out evaluation chips:", len(test_indices))


test_dataset = SatelliteDataset(
    CHIP_DIR,
    LABEL_FILE,
    test_indices
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# Model loader
# ============================================================

def load_model(model_path):

    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES
    )


    state_dict = torch.load(
        model_path,
        map_location=device
    )


    model.load_state_dict(
        state_dict
    )

    model = model.to(device)

    model.eval()

    return model


# ============================================================
# IoU calculation
# ============================================================

def compute_iou(
    predictions,
    targets,
    num_classes
):

    ious = []

    for class_id in range(
        num_classes
    ):

        prediction_class = (
            predictions == class_id
        )

        target_class = (
            targets == class_id
        )


        intersection = np.logical_and(
            prediction_class,
            target_class
        ).sum()


        union = np.logical_or(
            prediction_class,
            target_class
        ).sum()


        if union == 0:

            iou = np.nan

        else:

            iou = (
                intersection /
                union
            )


        ious.append(iou)


    return np.array(
        ious
    )


# ============================================================
# Evaluate model
# ============================================================

def evaluate_model(
    model,
    loader
):

    all_predictions = []
    all_targets = []


    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            outputs = model(
                images
            )


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            predictions = (
                predictions
                .cpu()
                .numpy()
            )

            labels = (
                labels
                .cpu()
                .numpy()
            )


            # Ignore NoData pixels

            valid = (
                labels != 255
            )


            all_predictions.append(
                predictions[valid]
            )

            all_targets.append(
                labels[valid]
            )


    predictions = np.concatenate(
        all_predictions
    )

    targets = np.concatenate(
        all_targets
    )


    ious = compute_iou(
        predictions,
        targets,
        NUM_CLASSES
    )


    mean_iou = np.nanmean(
        ious
    )


    accuracy = (
        predictions == targets
    ).mean() * 100


    cm = confusion_matrix(
        targets,
        predictions,
        labels=np.arange(
            NUM_CLASSES
        )
    )


    return (
        predictions,
        targets,
        ious,
        mean_iou,
        accuracy,
        cm
    )


# ============================================================
# Evaluate U-Net
# ============================================================

print()
print("=" * 70)
print("EVALUATING U-NET")
print("=" * 70)


unet = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=IN_CHANNELS,
    classes=NUM_CLASSES
)


unet_state = torch.load(
    UNET_MODEL,
    map_location=device
)


unet.load_state_dict(
    unet_state
)

unet = unet.to(device)

unet.eval()


(
    unet_predictions,
    unet_targets,
    unet_ious,
    unet_mean_iou,
    unet_accuracy,
    unet_cm
) = evaluate_model(
    unet,
    test_loader
)


print()
print("U-Net Accuracy:")
print(
    f"{unet_accuracy:.2f}%"
)

print()
print("U-Net Mean IoU:")
print(
    f"{unet_mean_iou:.4f}"
)

print()
print("U-Net Per-Class IoU:")

for name, value in zip(
    CLASS_NAMES,
    unet_ious
):

    print(
        f"{name}: {value:.4f}"
    )


# ============================================================
# Evaluate Foundation Model
# ============================================================

print()
print("=" * 70)
print("EVALUATING FOUNDATION MODEL")
print("=" * 70)


foundation = load_model(
    FOUNDATION_MODEL
)


(
    foundation_predictions,
    foundation_targets,
    foundation_ious,
    foundation_mean_iou,
    foundation_accuracy,
    foundation_cm
) = evaluate_model(
    foundation,
    test_loader
)


print()
print("Foundation Model Accuracy:")
print(
    f"{foundation_accuracy:.2f}%"
)

print()
print("Foundation Model Mean IoU:")
print(
    f"{foundation_mean_iou:.4f}"
)

print()
print("Foundation Model Per-Class IoU:")

for name, value in zip(
    CLASS_NAMES,
    foundation_ious
):

    print(
        f"{name}: {value:.4f}"
    )


# ============================================================
# Confusion Matrix - U-Net
# ============================================================

plt.figure(
    figsize=(8, 7)
)

sns.heatmap(
    unet_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.title(
    "U-Net Confusion Matrix - Guwahati"
)

plt.tight_layout()

unet_cm_file = os.path.join(
    OUTPUT_DIR,
    "unet_confusion_matrix.png"
)

plt.savefig(
    unet_cm_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Confusion Matrix - Foundation Model
# ============================================================

plt.figure(
    figsize=(8, 7)
)

sns.heatmap(
    foundation_cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.title(
    "Foundation Model Confusion Matrix - Guwahati"
)

plt.tight_layout()

foundation_cm_file = os.path.join(
    OUTPUT_DIR,
    "foundation_confusion_matrix.png"
)

plt.savefig(
    foundation_cm_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Classification reports
# ============================================================

unet_report = classification_report(
    unet_targets,
    unet_predictions,
    labels=np.arange(NUM_CLASSES),
    target_names=CLASS_NAMES,
    zero_division=0
)


foundation_report = classification_report(
    foundation_targets,
    foundation_predictions,
    labels=np.arange(NUM_CLASSES),
    target_names=CLASS_NAMES,
    zero_division=0
)


print()
print("=" * 70)
print("U-NET CLASSIFICATION REPORT")
print("=" * 70)

print(unet_report)


print()
print("=" * 70)
print("FOUNDATION MODEL CLASSIFICATION REPORT")
print("=" * 70)

print(foundation_report)


# ============================================================
# Save evaluation results
# ============================================================

results = {

    "study_area": "Guwahati",

    "evaluation_chips": len(
        test_indices
    ),

    "unet": {

        "accuracy_percent":
            float(unet_accuracy),

        "mean_iou":
            float(unet_mean_iou),

        "per_class_iou": {
            name: (
                None
                if np.isnan(value)
                else float(value)
            )
            for name, value in zip(
                CLASS_NAMES,
                unet_ious
            )
        }

    },

    "foundation_model": {

        "accuracy_percent":
            float(foundation_accuracy),

        "mean_iou":
            float(foundation_mean_iou),

        "per_class_iou": {
            name: (
                None
                if np.isnan(value)
                else float(value)
            )
            for name, value in zip(
                CLASS_NAMES,
                foundation_ious
            )
        }

    }

}


results_file = os.path.join(
    OUTPUT_DIR,
    "model_evaluation_results.json"
)


with open(
    results_file,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ============================================================
# Final output
# ============================================================

print()
print("=" * 70)
print("EXERCISE 4.2 EVALUATION COMPLETE")
print("=" * 70)

print()
print(
    "U-Net Mean IoU:",
    f"{unet_mean_iou:.4f}"
)

print(
    "Foundation Model Mean IoU:",
    f"{foundation_mean_iou:.4f}"
)

print()
print(
    "U-Net confusion matrix:"
)

print(unet_cm_file)

print()
print(
    "Foundation-model confusion matrix:"
)

print(foundation_cm_file)

print()
print(
    "Evaluation results:"
)

print(results_file)

print()
print("=" * 70)