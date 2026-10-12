import os
import json
import re
import numpy as np
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================
# GeoSense Phase 2 - Independent Test Experiment
# U-Net Semantic Segmentation
# Study Area: Guwahati
# ============================================================

YEAR = 2023

CHIP_DIR = r"data\satellite\colab_small\chips\2023"
LABEL_FILE = r"data\satellite\labels\labels_2023.npy"

MODEL_DIR = r"models\saved"
EVAL_DIR = r"models\evaluation"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 0.001

NUM_CLASSES = 5
IN_CHANNELS = 6

RANDOM_SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("GeoSense Phase 2 - Independent Test Experiment")
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

        self.chip_paths = sorted([
            os.path.join(chip_dir, f)
            for f in os.listdir(chip_dir)
            if f.lower().endswith(".tif")
        ])

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

        with rasterio.open(chip_path) as src:

            chip = src.read(
                indexes=[1, 2, 3, 4, 5, 6]
            ).astype(np.float32)

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

        chip_hwc = np.transpose(
            chip,
            (1, 2, 0)
        )

        chip_hwc = np.nan_to_num(
            chip_hwc,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )

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

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    ToTensorV2()
])

eval_transform = A.Compose([
    ToTensorV2()
])


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
# INDEPENDENT 60 / 10 / 10 SPLIT
# ============================================================

torch.manual_seed(RANDOM_SEED)

total = len(full_dataset)

if total != 80:
    print()
    print(
        "WARNING: Expected 80 chips, found:",
        total
    )

n_train = int(total * 0.75)
n_val = int(total * 0.125)
n_test = total - n_train - n_val

generator = torch.Generator().manual_seed(
    RANDOM_SEED
)

indices = torch.randperm(
    total,
    generator=generator
).tolist()

train_indices = indices[:n_train]

val_indices = indices[
    n_train:n_train + n_val
]

test_indices = indices[
    n_train + n_val:
]


print()
print("=" * 70)
print("INDEPENDENT DATA SPLIT")
print("=" * 70)

print("Total chips       :", total)
print("Training chips    :", len(train_indices))
print("Validation chips  :", len(val_indices))
print("Independent test  :", len(test_indices))

print()
print("Training indices   :", train_indices)
print("Validation indices :", val_indices)
print("Test indices       :", test_indices)


# ============================================================
# SUBSET DATASET
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
    eval_transform
)

test_dataset = SubsetWithTransform(
    full_dataset,
    test_indices,
    eval_transform
)


print()
print("Training dataset   :", len(train_dataset))
print("Validation dataset :", len(val_dataset))
print("Test dataset       :", len(test_dataset))


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

test_loader = DataLoader(
    test_dataset,
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
# LOSS AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=255
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

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
print("STARTING INDEPENDENT-TEST TRAINING")
print("=" * 70)

for epoch in range(NUM_EPOCHS):

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


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss = 0.0
    correct = 0
    total_valid = 0

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

            total_valid += (
                valid_pixels.sum().item()
            )

    val_loss /= len(val_loader)

    if total_valid > 0:

        val_accuracy = (
            correct / total_valid
        ) * 100

    else:

        val_accuracy = 0.0


    scheduler.step()


    history["train_loss"].append(
        train_loss
    )

    history["val_loss"].append(
        val_loss
    )

    history["val_accuracy"].append(
        val_accuracy
    )


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


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_model_path = os.path.join(
            MODEL_DIR,
            "unet_independent_test_best.pth"
        )

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print(
            "Best independent-test experiment "
            "model saved."
        )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "unet_independent_test_final.pth"
)

torch.save(
    model.state_dict(),
    final_model_path
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

best_model_path = os.path.join(
    MODEL_DIR,
    "unet_independent_test_best.pth"
)

model.load_state_dict(
    torch.load(
        best_model_path,
        map_location=DEVICE
    )
)

model.eval()


# ============================================================
# INDEPENDENT TEST EVALUATION
# ============================================================

print()
print("=" * 70)
print("INDEPENDENT TEST SET EVALUATION")
print("=" * 70)

correct = 0
total_valid = 0

class_correct = np.zeros(
    NUM_CLASSES,
    dtype=np.int64
)

class_total = np.zeros(
    NUM_CLASSES,
    dtype=np.int64
)

confusion = np.zeros(
    (NUM_CLASSES, NUM_CLASSES),
    dtype=np.int64
)

with torch.no_grad():

    for images, masks in test_loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        valid_pixels = (
            masks != 255
        )

        pred_valid = (
            predictions[valid_pixels]
            .cpu()
            .numpy()
        )

        true_valid = (
            masks[valid_pixels]
            .cpu()
            .numpy()
        )

        correct += np.sum(
            pred_valid == true_valid
        )

        total_valid += len(
            true_valid
        )

        for true_class, pred_class in zip(
            true_valid,
            pred_valid
        ):

            true_class = int(true_class)
            pred_class = int(pred_class)

            if (
                0 <= true_class < NUM_CLASSES
                and 0 <= pred_class < NUM_CLASSES
            ):

                confusion[
                    true_class,
                    pred_class
                ] += 1

                class_total[
                    true_class
                ] += 1

                if true_class == pred_class:

                    class_correct[
                        true_class
                    ] += 1


if total_valid > 0:

    test_accuracy = (
        correct / total_valid
    ) * 100

else:

    test_accuracy = 0.0


# ============================================================
# IOU
# ============================================================

iou = []

for class_id in range(NUM_CLASSES):

    intersection = confusion[
        class_id,
        class_id
    ]

    union = (
        confusion[class_id, :].sum()
        + confusion[:, class_id].sum()
        - intersection
    )

    if union > 0:

        class_iou = (
            intersection / union
        )

    else:

        class_iou = 0.0

    iou.append(
        float(class_iou)
    )


mean_iou = float(
    np.mean(iou)
)


class_names = [
    "Urban",
    "Vegetation",
    "Water",
    "Bare Land",
    "Agriculture"
]


print()
print("Independent Test Accuracy:")
print(
    f"{test_accuracy:.2f}%"
)

print()
print("Per-Class IoU:")

for name, value in zip(
    class_names,
    iou
):

    print(
        f"{name:15s}: {value:.4f}"
    )

print()
print(
    "Mean IoU:",
    f"{mean_iou:.4f}"
)

print()
print("Confusion Matrix:")

print(confusion)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "dataset": {
        "total_chips": total,
        "training_chips": len(train_indices),
        "validation_chips": len(val_indices),
        "independent_test_chips": len(test_indices)
    },

    "split_indices": {
        "train": train_indices,
        "validation": val_indices,
        "test": test_indices
    },

    "test_accuracy_percent":
        float(test_accuracy),

    "per_class_iou": {
        name: value
        for name, value in zip(
            class_names,
            iou
        )
    },

    "mean_iou":
        mean_iou,

    "confusion_matrix":
        confusion.tolist(),

    "training_history":
        history
}


results_path = os.path.join(
    EVAL_DIR,
    "independent_test_results.json"
)

with open(
    results_path,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("INDEPENDENT TEST EVALUATION COMPLETE")
print("=" * 70)

print()
print("Best model:")
print(best_model_path)

print()
print("Final model:")
print(final_model_path)

print()
print("Results:")
print(results_path)

print()
print(
    "Independent test accuracy:",
    f"{test_accuracy:.2f}%"
)

print(
    "Independent test mIoU:",
    f"{mean_iou:.4f}"
)

print()
print("=" * 70)