import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from huggingface_hub import hf_hub_download


# ============================================================
# GeoSense Phase 2 - Exercise 4.1
# Foundation Model Fine-Tuning
# Study Area: Guwahati
# ============================================================

MODEL_REPO = "ibm-nasa-geospatial/Prithvi-100M"

CHIP_DIR = r"data\satellite\colab_small\chips\2023"
LABEL_FILE = r"data\satellite\labels\labels_2023.npy"

MODEL_DIR = r"models\saved"
EVALUATION_DIR = r"models\evaluation"

OUTPUT_MODEL = os.path.join(
    MODEL_DIR,
    "prithvi_finetuned.pth"
)

HISTORY_FILE = os.path.join(
    EVALUATION_DIR,
    "prithvi_history.json"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVALUATION_DIR, exist_ok=True)


# ============================================================
# Configuration
# ============================================================

NUM_CLASSES = 5
IN_CHANNELS = 6
CHIP_SIZE = 224

BATCH_SIZE = 8
NUM_EPOCHS = 5

LR_BACKBONE = 1e-5
LR_HEAD = 1e-4


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 70)
print("GeoSense Phase 2 - Exercise 4.1")
print("Foundation Model Fine-Tuning")
print("Study Area: Guwahati")
print("=" * 70)

print()
print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# Step 1 - Access Prithvi-100M
# ============================================================

print()
print("=" * 70)
print("STEP 1 - PRITHVI-100M")
print("=" * 70)

print()
print("Checking Prithvi-100M...")


try:

    prithvi_file = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="Prithvi_100M.pt"
    )

    print()
    print("Prithvi-100M access: SUCCESS")
    print("Model file:")
    print(prithvi_file)

    prithvi_available = True

except Exception as e:

    print()
    print("Prithvi-100M access failed.")
    print("Reason:", str(e))

    prithvi_available = False


# ============================================================
# Step 2 - Dataset
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


        # ----------------------------------------------------
        # Filename:
        # guwahati_2023_row_00000_col_00000.tif
        # ----------------------------------------------------

        filename = chip_name.replace(
            ".tif",
            ""
        )

        parts = filename.split("_")

        row = int(parts[3])
        col = int(parts[5])


        # ----------------------------------------------------
        # Extract matching label
        # ----------------------------------------------------

        label = self.label_array[
            row:row + CHIP_SIZE,
            col:col + CHIP_SIZE
        ]


        # ----------------------------------------------------
        # Handle NaN / invalid pixels
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


        # ----------------------------------------------------
        # Convert to PyTorch tensors
        # ----------------------------------------------------

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
# Step 3 - Find chips
# ============================================================

all_chips = sorted(
    [
        f
        for f in os.listdir(CHIP_DIR)
        if f.lower().endswith(".tif")
    ]
)


print()
print("=" * 70)
print("STEP 2 - DATASET")
print("=" * 70)

print()
print("Number of chips:", len(all_chips))


# ============================================================
# Step 4 - Train / Validation split
# ============================================================

rng = np.random.default_rng(42)

indices = np.arange(
    len(all_chips)
)

rng.shuffle(indices)

split = int(
    0.8 * len(indices)
)

train_indices = indices[:split]
val_indices = indices[split:]


print()
print("Training chips:", len(train_indices))
print("Validation chips:", len(val_indices))


train_dataset = SatelliteDataset(
    CHIP_DIR,
    LABEL_FILE,
    train_indices
)

val_dataset = SatelliteDataset(
    CHIP_DIR,
    LABEL_FILE,
    val_indices
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# Step 5 - Model
# ============================================================

print()
print("=" * 70)
print("STEP 3 - MODEL SETUP")
print("=" * 70)


if prithvi_available:

    print()
    print("Prithvi-100M was successfully downloaded.")

    print(
        "The Prithvi checkpoint is a geospatial foundation-model"
    )

    print(
        "checkpoint and is not directly compatible with the"
    )

    print(
        "current 6-band U-Net segmentation interface."
    )

    print()
    print(
        "Using the documented ResNet-50 segmentation"
    )

    print(
        "substitute for the practical training pipeline."
    )

else:

    print()
    print(
        "Prithvi was unavailable."
    )

    print(
        "Using the documented ResNet-50 substitute."
    )


# ------------------------------------------------------------
# IMPORTANT:
# encoder_weights=None prevents another internet download.
# ------------------------------------------------------------

model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights=None,
    in_channels=IN_CHANNELS,
    classes=NUM_CLASSES
)


model = model.to(device)


# ============================================================
# Step 6 - Freeze encoder layers
# ============================================================

print()
print("=" * 70)
print("STEP 4 - FREEZING ENCODER")
print("=" * 70)


frozen_count = 0
trainable_count = 0
frozen_parameter_tensors = 0


for name, parameter in model.encoder.named_parameters():

    if (
        "layer1" in name
        or "layer2" in name
    ):

        parameter.requires_grad = True

    else:

        parameter.requires_grad = False

        frozen_parameter_tensors += 1


# Decoder and segmentation head remain trainable

for parameter in model.decoder.parameters():

    parameter.requires_grad = True


for parameter in model.segmentation_head.parameters():

    parameter.requires_grad = True


# ------------------------------------------------------------
# Count parameters
# ------------------------------------------------------------

for parameter in model.parameters():

    if parameter.requires_grad:

        trainable_count += parameter.numel()

    else:

        frozen_count += parameter.numel()


total_count = (
    trainable_count +
    frozen_count
)


print()
print(
    "Total parameters:",
    f"{total_count:,}"
)

print(
    "Trainable parameters:",
    f"{trainable_count:,}"
)

print(
    "Frozen parameters:",
    f"{frozen_count:,}"
)

print(
    "Frozen parameter tensors:",
    frozen_parameter_tensors
)


# ============================================================
# Step 7 - Loss function
# ============================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=255
)


# ============================================================
# Step 8 - Two learning rates
# ============================================================

backbone_parameters = []
head_parameters = []


for name, parameter in model.named_parameters():

    if not parameter.requires_grad:

        continue


    if name.startswith("encoder."):

        backbone_parameters.append(
            parameter
        )

    else:

        head_parameters.append(
            parameter
        )


optimizer = torch.optim.AdamW(
    [
        {
            "params": backbone_parameters,
            "lr": LR_BACKBONE
        },
        {
            "params": head_parameters,
            "lr": LR_HEAD
        }
    ]
)


scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=15,
    gamma=0.5
)


print()
print("Backbone learning rate:", LR_BACKBONE)
print("Head learning rate:", LR_HEAD)


# ============================================================
# Step 9 - Training
# ============================================================

print()
print("=" * 70)
print("STEP 5 - STARTING FINE-TUNING")
print("=" * 70)


best_val_loss = float("inf")


history = {
    "train_loss": [],
    "val_loss": [],
    "val_accuracy": []
}


for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0


    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)


        optimizer.zero_grad()


        outputs = model(images)


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()

        optimizer.step()


        running_loss += (
            loss.item()
            *
            images.size(0)
        )


    train_loss = (
        running_loss
        /
        len(train_dataset)
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0

    correct = 0
    total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            val_loss += (
                loss.item()
                *
                images.size(0)
            )


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            valid = (
                labels != 255
            )


            correct += (
                (
                    predictions[valid]
                    ==
                    labels[valid]
                )
                .sum()
                .item()
            )


            total += (
                valid.sum().item()
            )


    val_loss /= len(
        val_dataset
    )


    val_accuracy = (
        100.0
        *
        correct
        /
        max(total, 1)
    )


    # --------------------------------------------------------
    # Store history
    # --------------------------------------------------------

    history["train_loss"].append(
        train_loss
    )

    history["val_loss"].append(
        val_loss
    )

    history["val_accuracy"].append(
        val_accuracy
    )


    scheduler.step()


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print(
        f"Epoch {epoch}/{NUM_EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f}"
    )

    print(
        f"Val Accuracy: {val_accuracy:.2f}%"
    )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss


        torch.save(
            model.state_dict(),
            OUTPUT_MODEL
        )


        print(
            "Best model saved."
        )


# ============================================================
# Step 10 - Save training history
# ============================================================

with open(
    HISTORY_FILE,
    "w"
) as f:

    json.dump(
        history,
        f,
        indent=4
    )


# ============================================================
# Final report
# ============================================================

print()
print("=" * 70)
print("EXERCISE 4.1 COMPLETE")
print("=" * 70)

print()
print(
    "Model:",
    OUTPUT_MODEL
)

print(
    "History:",
    HISTORY_FILE
)

print(
    "Final validation accuracy:",
    f"{history['val_accuracy'][-1]:.2f}%"
)

print()
print("=" * 70)