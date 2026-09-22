import os
import json
import re
import numpy as np
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================
# GeoSense Phase 2 - Exercise 3.2
# U-Net Semantic Segmentation
# Study Area: Guwahati
# ============================================================

YEAR = 2023

# IMPORTANT:
# Use the SMALL 80-chip dataset for CPU training.
CHIP_DIR = r"data\satellite\colab_small\chips\2023"

# Original full-year label file
LABEL_FILE = r"data\satellite\labels\labels_2023.npy"

MODEL_DIR = r"models\saved"
EVAL_DIR = r"models\evaluation"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

# Training settings
BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 0.001

NUM_CLASSES = 5
IN_CHANNELS = 6
RANDOM_SEED = 42


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print()
print("=" * 70)
print("GeoSense Phase 2 - Exercise 3.2")
print("U-Net Semantic Segmentation")
print("Study Area: Guwahati")
print("=" * 70)

print()
print("PyTorch version:", torch.__version__)
print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU not available - using CPU")


# ============================================================
# DATASET
# ============================================================

class SatelliteDataset(Dataset):

    def __init__(self, chip_dir, label_file, transform=None):

        self.chip_dir = chip_dir
        self.transform = transform

        self.chip_paths = sorted(
            [
                os.path.join(chip_dir, f)
                for f in os.listdir(chip_dir)
                if f.lower().endswith(".tif")
            ]
        )

        if len(self.chip_paths) == 0:
            raise RuntimeError(
                f"No TIFF chips found in {chip_dir}"
            )

        print()
        print("Number of chips:", len(self.chip_paths))

        self.label_arr = np.load(
            label_file,
            mmap_mode="r"
        )

        print(
            "Label size:",
            self.label_arr.shape
        )

    def __len__(self):
        return len(self.chip_paths)

    def __getitem__(self, idx):

        chip_path = self.chip_paths[idx]

        # ----------------------------------------------------
        # Read first 6 Sentinel-2 bands
        # B2, B3, B4, B8, B11, B12
        # ----------------------------------------------------

        with rasterio.open(chip_path) as src:

            chip = src.read(
                indexes=[1, 2, 3, 4, 5, 6]
            ).astype(np.float32)

        # ----------------------------------------------------
        # Extract row and column from filename
        # Example:
        # guwahati_2023_row_00000_col_00000.tif
        # ----------------------------------------------------

        filename = os.path.basename(chip_path)

        match = re.search(
            r"row_(\d+)_col_(\d+)",
            filename
        )

        if match is None:
            raise RuntimeError(
                f"Could not determine row/column from filename: "
                f"{filename}"
            )

        row = int(match.group(1))
        col = int(match.group(2))

        chip_height = chip.shape[1]
        chip_width = chip.shape[2]

        # ----------------------------------------------------
        # Extract matching label patch
        # ----------------------------------------------------

        label = self.label_arr[
            row:row + chip_height,
            col:col + chip_width
        ]

        if label.shape != (
            chip_height,
            chip_width
        ):
            raise RuntimeError(
                f"Label shape {label.shape} does not match "
                f"chip shape {(chip_height, chip_width)} "
                f"for {filename}"
            )

        # ----------------------------------------------------
        # CHW -> HWC
        # ----------------------------------------------------

        chip_hwc = np.transpose(
            chip,
            (1, 2, 0)
        )

        # ----------------------------------------------------
        # Replace invalid values
        # ----------------------------------------------------

        chip_hwc = np.nan_to_num(
            chip_hwc,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )

        # ----------------------------------------------------
        # Apply augmentation
        # ----------------------------------------------------

        if self.transform is not None:

            augmented = self.transform(
                image=chip_hwc,
                mask=label.astype(np.int64)
            )

            chip_tensor = augmented["image"]
            label_tensor = augmented["mask"].long()

        else:

            chip_tensor = torch.tensor(
                chip_hwc,
                dtype=torch.float32
            ).permute(2, 0, 1)

            label_tensor = torch.tensor(
                label.astype(np.int64),
                dtype=torch.long
            )

        return chip_tensor, label_tensor


# ============================================================
# AUGMENTATION
# ============================================================

train_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        ToTensorV2()
    ]
)

val_transform = A.Compose(
    [
        ToTensorV2()
    ]
)


# ============================================================
# CREATE DATASET
# ============================================================

print()
print("-" * 70)
print("Creating dataset")
print("-" * 70)

full_dataset = SatelliteDataset(
    CHIP_DIR,
    LABEL_FILE,
    transform=None
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

torch.manual_seed(RANDOM_SEED)

n_train = int(
    len(full_dataset) * 0.8
)

n_val = len(full_dataset) - n_train

train_indices, val_indices = random_split(
    range(len(full_dataset)),
    [n_train, n_val],
    generator=torch.Generator().manual_seed(
        RANDOM_SEED
    )
)


# ============================================================
# SUBSET WITH TRANSFORM
# ============================================================

class SubsetWithTransform(Dataset):

    def __init__(
        self,
        dataset,
        indices,
        transform
    ):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        original_idx = self.indices[idx]

        old_transform = self.dataset.transform

        self.dataset.transform = self.transform

        item = self.dataset[original_idx]

        self.dataset.transform = old_transform

        return item


train_dataset = SubsetWithTransform(
    full_dataset,
    train_indices,
    train_transform
)

val_dataset = SubsetWithTransform(
    full_dataset,
    val_indices,
    val_transform
)

print()
print("Training chips:", len(train_dataset))
print("Validation chips:", len(val_dataset))


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# BUILD U-NET
# ============================================================

print()
print("-" * 70)
print("Building U-Net")
print("-" * 70)

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=IN_CHANNELS,
    classes=NUM_CLASSES
)

model = model.to(DEVICE)

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "Model parameters:",
    f"{parameter_count:,}"
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=255
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=15,
    gamma=0.5
)


# ============================================================
# TRAINING HISTORY
# ============================================================

history = {
    "train_loss": [],
    "val_loss": [],
    "val_accuracy": []
}

best_val_loss = float("inf")


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

for epoch in range(NUM_EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0

    for images, masks in train_loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            masks
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0

    correct = 0
    total = 0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                masks
            )

            val_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            valid_pixels = (
                masks != 255
            )

            correct += (
                (
                    predictions[valid_pixels]
                    == masks[valid_pixels]
                )
                .sum()
                .item()
            )

            total += (
                valid_pixels.sum().item()
            )

    val_loss /= len(val_loader)

    if total > 0:
        val_accuracy = (
            correct / total
        ) * 100
    else:
        val_accuracy = 0.0


    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler.step()


    # --------------------------------------------------------
    # Save history
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


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}]"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy:.2f}%"
    )

    print(
        f"Learning Rate: "
        f"{optimizer.param_groups[0]['lr']:.6f}"
    )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_model_path = os.path.join(
            MODEL_DIR,
            "unet_best.pth"
        )

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print("✓ Best model saved")


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "unet_final.pth"
)

torch.save(
    model.state_dict(),
    final_model_path
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_path = os.path.join(
    EVAL_DIR,
    "unet_history.json"
)

with open(
    history_path,
    "w"
) as f:

    json.dump(
        history,
        f,
        indent=4
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("U-NET TRAINING COMPLETE")
print("=" * 70)

print()
print("Best model:")
print(
    os.path.join(
        MODEL_DIR,
        "unet_best.pth"
    )
)

print()
print("Final model:")
print(final_model_path)

print()
print("Training history:")
print(history_path)

print()
print("Final validation accuracy:")
print(
    f"{history['val_accuracy'][-1]:.2f}%"
)

print()
print("=" * 70)