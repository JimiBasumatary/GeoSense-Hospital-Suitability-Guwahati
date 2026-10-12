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

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHIP_DIR = os.path.join(ROOT_DIR, "data", "satellite", "colab_small", "chips", "2023")
LABEL_FILE = os.path.join(ROOT_DIR, "data", "satellite", "labels", "labels_2023.npy")

UNET_MODEL = os.path.join(ROOT_DIR, "models", "saved", "unet_best.pth")
FOUNDATION_MODEL = os.path.join(ROOT_DIR, "models", "saved", "prithvi_finetuned.pth")

OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_CLASSES = 5
IN_CHANNELS = 6
CHIP_SIZE = 224
BATCH_SIZE = 8
RANDOM_SEED = 42

CLASS_NAMES = ["Urban", "Vegetation", "Water", "Bare Land", "Agriculture"]
CLASS_IDS = np.arange(NUM_CLASSES)
NODATA_LABEL = 255

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SatelliteDataset(Dataset):
    def __init__(self, chip_dir, label_file, indices):
        self.chip_dir = chip_dir
        self.label_array = np.load(label_file, mmap_mode="r")
        all_chips = sorted(
            f for f in os.listdir(chip_dir) if f.lower().endswith(".tif")
        )
        if len(all_chips) == 0:
            raise FileNotFoundError(f"No .tif chips found in: {chip_dir}")
        if max(indices, default=-1) >= len(all_chips):
            raise IndexError(
                f"Split contains an index outside the {len(all_chips)} available chips."
            )
        self.chips = [all_chips[int(i)] for i in indices]

    def __len__(self):
        return len(self.chips)

    def __getitem__(self, index):
        import rasterio

        chip_name = self.chips[index]
        chip_path = os.path.join(self.chip_dir, chip_name)

        with rasterio.open(chip_path) as src:
            image = src.read(indexes=[1, 2, 3, 4, 5, 6]).astype(np.float32)

        # Expected chip filename: guwahati_2023_row_00000_col_00000.tif
        stem = os.path.splitext(chip_name)[0]
        parts = stem.split("_")
        try:
            row = int(parts[3])
            col = int(parts[5])
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"Cannot parse row/column from chip filename: {chip_name}"
            ) from exc

        label = self.label_array[row:row + CHIP_SIZE, col:col + CHIP_SIZE]
        if label.shape != (CHIP_SIZE, CHIP_SIZE):
            raise ValueError(
                f"Label window for {chip_name} has shape {label.shape}, "
                f"expected {(CHIP_SIZE, CHIP_SIZE)}."
            )

        # Replace non-finite values band-by-band, then constrain input range.
        for band in range(image.shape[0]):
            band_data = image[band]
            valid = np.isfinite(band_data)
            if valid.any():
                band_data[~valid] = np.nanmean(band_data)
            else:
                band_data[:] = 0

        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        image = np.clip(image, 0, 1)

        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(label.copy(), dtype=torch.long),
        )


def load_unet(model_path, encoder_name):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES,
    )
    state = torch.load(model_path, map_location=device)
    # Supports either a plain state_dict or a common wrapped checkpoint.
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def compute_iou(predictions, targets):
    ious = []
    for class_id in CLASS_IDS:
        pred_class = predictions == class_id
        true_class = targets == class_id
        intersection = np.logical_and(pred_class, true_class).sum()
        union = np.logical_or(pred_class, true_class).sum()
        ious.append(np.nan if union == 0 else intersection / union)
    return np.asarray(ious, dtype=float)


def evaluate_model(model, loader):
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()

            # Exclude NoData (255) and any unexpected class values.
            valid = (labels != NODATA_LABEL) & (labels >= 0) & (labels < NUM_CLASSES)
            if np.any(valid):
                all_predictions.append(predictions[valid])
                all_targets.append(labels[valid])

    if not all_targets:
        raise ValueError(
            "No valid labelled pixels were found in the evaluation chips."
        )

    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)

    cm = confusion_matrix(targets, predictions, labels=CLASS_IDS)
    ious = compute_iou(predictions, targets)
    mean_iou = float(np.nanmean(ious))
    accuracy = float((predictions == targets).mean() * 100.0)
    return predictions, targets, ious, mean_iou, accuracy, cm


def save_confusion_matrix(cm, filename, title, cmap):
    # Row-normalise: each row is the percentage distribution for one actual class.
    row_totals = cm.sum(axis=1, keepdims=True)
    cm_percent = np.divide(
        cm.astype(float) * 100.0,
        row_totals,
        out=np.zeros_like(cm, dtype=float),
        where=row_totals != 0,
    )

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        cm_percent,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        vmin=0,
        vmax=100,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={"label": "Percentage of actual-class pixels (%)"},
    )
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    filename = os.path.abspath(os.fspath(filename))
    print("DEBUG filename:", repr(filename))
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved row-normalised confusion matrix: {filename}")
    print("Raw confusion matrix (counts):")
    print(cm)
    print("Row-normalised confusion matrix (%):")
    print(np.round(cm_percent, 2))


def main():
    print("=" * 70)
    print("GeoSense Phase 2 - Exercise 4.2")
    print("U-Net vs Foundation Model Evaluation - Guwahati")
    print("=" * 70)
    print("Device:", device)

    if not os.path.isdir(CHIP_DIR):
        raise FileNotFoundError(f"Chip directory not found: {CHIP_DIR}")
    if not os.path.isfile(LABEL_FILE):
        raise FileNotFoundError(f"Label array not found: {LABEL_FILE}")

    all_chips = sorted(
        f for f in os.listdir(CHIP_DIR) if f.lower().endswith(".tif")
    )
    if len(all_chips) < 2:
        raise ValueError(f"Need at least two chips; found {len(all_chips)}.")

    # IMPORTANT:
    # This is a reproducible 80/20 held-out split. It is not automatically the
    # validation split used during training. Only call it "validation" if these
    # indices match the split used to train/save the selected checkpoint.
    rng = np.random.default_rng(RANDOM_SEED)
    indices = np.arange(len(all_chips))
    rng.shuffle(indices)
    split = int(0.8 * len(indices))
    train_indices = indices[:split]
    heldout_indices = indices[split:]

    print(f"Total chips: {len(all_chips)}")
    print(f"Training portion (split only): {len(train_indices)}")
    print(f"Held-out evaluation chips: {len(heldout_indices)}")
    print("Split seed:", RANDOM_SEED)
    print("NoData label excluded:", NODATA_LABEL)

    dataset = SatelliteDataset(CHIP_DIR, LABEL_FILE, heldout_indices)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("\nEvaluating U-Net checkpoint...")
    # The original evaluation code specified ResNet-34 for unet_best.pth.
    unet = load_unet(UNET_MODEL, encoder_name="resnet34")
    unet_predictions, unet_targets, unet_ious, unet_miou, unet_accuracy, unet_cm = (
        evaluate_model(unet, loader)
    )

    unet_cm_file = os.path.join(OUTPUT_DIR, "unet_confusion_matrix_row_normalized.png")
    save_confusion_matrix(
        unet_cm,
        unet_cm_file,
        "U-Net Row-Normalised Confusion Matrix - Guwahati Held-out Evaluation",
        "Blues",
    )

    print("\nU-Net accuracy: {:.2f}%".format(unet_accuracy))
    print("U-Net mean IoU: {:.4f}".format(unet_miou))
    print("U-Net per-class IoU:")
    for name, value in zip(CLASS_NAMES, unet_ious):
        print(f"  {name}: {value:.4f}")

    # Keep the foundation-model comparison from the original script. This assumes
    # the supplied foundation checkpoint is an SMP U-Net with a ResNet-50 encoder.
    foundation_results = None
    if os.path.isfile(FOUNDATION_MODEL):
        print("\nEvaluating foundation-model checkpoint...")
        foundation = load_unet(FOUNDATION_MODEL, encoder_name="resnet50")
        fp, ft, f_ious, f_miou, f_accuracy, f_cm = evaluate_model(foundation, loader)
        foundation_cm_file = os.path.join(
            OUTPUT_DIR, "foundation_confusion_matrix_row_normalized.png"
        )
        save_confusion_matrix(
            f_cm,
            foundation_cm_file,
            "Foundation Model Row-Normalised Confusion Matrix - Guwahati Held-out Evaluation",
            "Greens",
        )
        print("\nFoundation-model accuracy: {:.2f}%".format(f_accuracy))
        print("Foundation-model mean IoU: {:.4f}".format(f_miou))
        print("Foundation-model per-class IoU:")
        for name, value in zip(CLASS_NAMES, f_ious):
            print(f"  {name}: {value:.4f}")
        foundation_results = {
            "accuracy_percent": f_accuracy,
            "mean_iou": f_miou,
            "per_class_iou": {
                name: (None if np.isnan(value) else float(value))
                for name, value in zip(CLASS_NAMES, f_ious)
            },
        }
    else:
        print("\nFoundation checkpoint not found; skipping foundation-model evaluation.")


    report = classification_report(
        unet_targets,
        unet_predictions,
        labels=CLASS_IDS,
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    print("\nU-NET CLASSIFICATION REPORT\n" + report)

    foundation_report = None
    if foundation_results is not None:
        foundation_report = classification_report(
            ft,
            fp,
            labels=CLASS_IDS,
            target_names=CLASS_NAMES,
            zero_division=0,
        )
        print("\nRESNET-50 U-NET SUBSTITUTE CLASSIFICATION REPORT\n")
        print(foundation_report)

    # Convert classification reports to dictionaries so Table 3 metrics can be saved.
    unet_report_dict = classification_report(
        unet_targets,
        unet_predictions,
        labels=CLASS_IDS,
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )

    foundation_report_dict = None
    if foundation_results is not None:
        foundation_report_dict = classification_report(
            ft,
            fp,
            labels=CLASS_IDS,
            target_names=CLASS_NAMES,
            zero_division=0,
            output_dict=True,
        )
        foundation_results["macro_precision"] = float(
            foundation_report_dict["macro avg"]["precision"]
        )
        foundation_results["macro_recall"] = float(
            foundation_report_dict["macro avg"]["recall"]
        )
        foundation_results["macro_f1"] = float(
            foundation_report_dict["macro avg"]["f1-score"]
        )

    results = {
        "study_area": "Guwahati",
        "evaluation_type": (
            "reproducible_80_20_held_out_split_"
            "not_verified_training_validation_split"
        ),
        "random_seed": RANDOM_SEED,
        "total_chips": len(all_chips),
        "training_portion_chips": len(train_indices),
        "heldout_evaluation_chips": len(heldout_indices),
        "labelled_pixels_evaluated": int(unet_targets.size),
        "nodata_label_excluded": NODATA_LABEL,
        "unet": {
            "accuracy_percent": unet_accuracy,
            "mean_iou": unet_miou,
            "macro_precision": float(unet_report_dict["macro avg"]["precision"]),
            "macro_recall": float(unet_report_dict["macro avg"]["recall"]),
            "macro_f1": float(unet_report_dict["macro avg"]["f1-score"]),
            "per_class_iou": {
                name: (
                    None if np.isnan(value) else float(value)
                )
                for name, value in zip(CLASS_NAMES, unet_ious)
            },
            "confusion_matrix_counts": unet_cm.tolist(),
        },
        "foundation_model": foundation_results,
    }

    results_file = os.path.join(
        OUTPUT_DIR, "model_evaluation_results.json"
    )
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    # Create a concise model-comparison summary.
    summary_file = os.path.join(ROOT_DIR, "outputs", "model_comparison.md")
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)

    summary = [
        "# Guwahati Model Evaluation Summary",
        "",
        f"- Total chips: {len(all_chips)}",
        f"- Held-out evaluation chips: {len(heldout_indices)}",
        f"- Labelled pixels evaluated: {unet_targets.size:,}",
        f"- Random seed: {RANDOM_SEED}",
        "- Evaluation split: reproducible 80/20 split; "
        "not confirmed as the training validation split",
        "",
        "| Metric | U-Net | ResNet-50 U-Net substitute |",
        "|---|---:|---:|",
        f"| Accuracy (%) | {unet_accuracy:.2f} | "
        f"{foundation_results['accuracy_percent']:.2f} |"
        if foundation_results is not None
        else f"| Accuracy (%) | {unet_accuracy:.2f} | Not evaluated |",
        f"| Mean IoU (%) | {unet_miou * 100:.2f} | "
        f"{foundation_results['mean_iou'] * 100:.2f} |"
        if foundation_results is not None
        else f"| Mean IoU (%) | {unet_miou * 100:.2f} | Not evaluated |",
        f"| Macro precision (%) | {unet_report_dict['macro avg']['precision'] * 100:.2f} | "
        f"{foundation_results['macro_precision'] * 100:.2f} |"
        if foundation_results is not None
        else f"| Macro precision (%) | {unet_report_dict['macro avg']['precision'] * 100:.2f} | Not evaluated |",
        f"| Macro recall (%) | {unet_report_dict['macro avg']['recall'] * 100:.2f} | "
        f"{foundation_results['macro_recall'] * 100:.2f} |"
        if foundation_results is not None
        else f"| Macro recall (%) | {unet_report_dict['macro avg']['recall'] * 100:.2f} | Not evaluated |",
        f"| Macro F1-score (%) | {unet_report_dict['macro avg']['f1-score'] * 100:.2f} | "
        f"{foundation_results['macro_f1'] * 100:.2f} |"
        if foundation_results is not None
        else f"| Macro F1-score (%) | {unet_report_dict['macro avg']['f1-score'] * 100:.2f} | Not evaluated |",
        "",
        "## U-Net Classification Report",
        "",
        "```text",
        report,
        "```",
    ]

    if foundation_report is not None:
        summary.extend([
            "",
            "## ResNet-50 U-Net Substitute Classification Report",
            "",
            "```text",
            foundation_report,
            "```",
        ])

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")

    print("\n" + "=" * 70)
    print("EXERCISE 4.2 COMPLETE")
    print("=" * 70)
    print("Evaluation results saved:", os.path.abspath(results_file))
    print("Model comparison summary saved:", os.path.abspath(summary_file))
    print("U-Net confusion matrix:", os.path.abspath(unet_cm_file))

    if foundation_results is not None:
        print(
            "ResNet-50 U-Net substitute confusion matrix:",
            os.path.abspath(foundation_cm_file),
        )

    print("Total chips:", len(all_chips))
    print("Held-out evaluation chips:", len(heldout_indices))
    print("Labelled pixels evaluated:", f"{unet_targets.size:,}")
    print("U-Net accuracy: {:.2f}%".format(unet_accuracy))
    print("U-Net mean IoU: {:.4f}".format(unet_miou))

    if foundation_results is not None:
        print(
            "ResNet-50 U-Net substitute accuracy: "
            "{:.2f}%".format(foundation_results["accuracy_percent"])
        )
        print(
            "ResNet-50 U-Net substitute mean IoU: "
            "{:.4f}".format(foundation_results["mean_iou"])
        )

    print("\nGenerated evaluation files:")
    print("1.", os.path.abspath(unet_cm_file))
    if foundation_results is not None:
        print("2.", os.path.abspath(foundation_cm_file))
    print("3.", os.path.abspath(results_file))
    print("4.", os.path.abspath(summary_file))

    print("\nNOTE: This is a held-out evaluation split, not necessarily")
    print("the validation split used during checkpoint training.")
    print("=" * 70)

if __name__ == "__main__":
    main()
