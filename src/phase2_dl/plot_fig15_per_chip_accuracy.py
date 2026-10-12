
import os
import json
import numpy as np
import torch
import rasterio
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

CHIP_DIR = os.path.join(
    ROOT, "data", "satellite", "colab_small", "chips", "2023"
)
LABEL_FILE = os.path.join(
    ROOT, "data", "satellite", "labels", "labels_2023.npy"
)
UNET_FILE = os.path.join(ROOT, "models", "saved", "unet_best.pth")
SUBSTITUTE_FILE = os.path.join(
    ROOT, "models", "saved", "prithvi_finetuned.pth"
)
OUT_DIR = os.path.join(ROOT, "outputs", "plots")
os.makedirs(OUT_DIR, exist_ok=True)

CHIP_SIZE = 224
NUM_CLASSES = 5
NODATA = 255
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path, encoder):
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=6,
        classes=NUM_CLASSES,
    )
    state = torch.load(path, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def load_chip(path):
    with rasterio.open(path) as src:
        image = src.read(indexes=[1, 2, 3, 4, 5, 6]).astype(np.float32)

    for band in range(image.shape[0]):
        values = image[band]
        valid = np.isfinite(values)
        if valid.any():
            values[~valid] = np.nanmean(values)
        else:
            values[:] = 0

    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    image = np.clip(image, 0, 1)
    return torch.tensor(image, dtype=torch.float32).unsqueeze(0)


def main():
    print("Device:", DEVICE)
    print("Loading labels and held-out chip list...")

    labels = np.load(LABEL_FILE, mmap_mode="r")
    chips = sorted(
        f for f in os.listdir(CHIP_DIR) if f.lower().endswith(".tif")
    )

    if len(chips) != 80:
        print("WARNING: Expected 80 chips, found", len(chips))

    rng = np.random.default_rng(SEED)
    indices = np.arange(len(chips))
    rng.shuffle(indices)
    split = int(0.8 * len(indices))
    heldout_indices = indices[split:]

    # Match the sorted-chip, seed-42 80/20 split used by evaluation_models.py.
    unet = load_model(UNET_FILE, "resnet34")
    substitute = load_model(SUBSTITUTE_FILE, "resnet50")

    rows = []
    print("Calculating accuracy for each held-out chip...")

    with torch.no_grad():
        for index in heldout_indices:
            chip_name = chips[int(index)]
            stem = os.path.splitext(chip_name)[0]
            parts = stem.split("_")

            try:
                row = int(parts[3])
                col = int(parts[5])
            except (IndexError, ValueError):
                raise ValueError(
                    "Unexpected chip filename format: " + chip_name
                )

            target = np.asarray(
                labels[row:row + CHIP_SIZE, col:col + CHIP_SIZE]
            )

            if target.shape != (CHIP_SIZE, CHIP_SIZE):
                print("Skipping incomplete label chip:", chip_name)
                continue

            valid = (
                (target != NODATA)
                & (target >= 0)
                & (target < NUM_CLASSES)
            )
            valid_count = int(valid.sum())

            if valid_count == 0:
                print("Skipping chip with no valid labels:", chip_name)
                continue

            image = load_chip(os.path.join(CHIP_DIR, chip_name)).to(DEVICE)

            pred_unet = torch.argmax(unet(image), dim=1)[0].cpu().numpy()
            pred_sub = torch.argmax(substitute(image), dim=1)[0].cpu().numpy()

            acc_unet = 100.0 * np.mean(pred_unet[valid] == target[valid])
            acc_sub = 100.0 * np.mean(pred_sub[valid] == target[valid])

            rows.append({
                "chip": chip_name,
                "valid_pixels": valid_count,
                "unet_accuracy_percent": float(acc_unet),
                "substitute_accuracy_percent": float(acc_sub),
            })

            print(
                f"{chip_name}: U-Net={acc_unet:.2f}%, "
                f"substitute={acc_sub:.2f}%"
            )

    if not rows:
        raise RuntimeError("No chips had valid labelled pixels.")

    # Save the actual per-chip measurements.
    csv_path = os.path.join(OUT_DIR, "fig15_per_chip_accuracy.csv")
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(OUT_DIR, "fig15_per_chip_accuracy.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    x = [r["unet_accuracy_percent"] for r in rows]
    y = [r["substitute_accuracy_percent"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(x, y, s=65, alpha=0.85)

    # Equality line: points above it favour the substitute for that chip.
    ax.plot([0, 100], [0, 100], linestyle="--", linewidth=1.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("U-Net pixel accuracy (%)")
    ax.set_ylabel("ResNet-50 U-Net substitute pixel accuracy (%)")
    ax.set_title("Per-Chip Pixel Accuracy — Guwahati Held-out Chips")
    ax.grid(True, alpha=0.25)

    fig.text(
        0.5, 0.015,
        f"Valid chips: {len(rows)} | Split seed: {SEED} | "
        "NoData label 255 excluded",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    plot_path = os.path.join(
        OUT_DIR, "fig15_per_chip_accuracy_comparison.png"
    )
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nFIG. 15 GENERATED")
    print("Valid chips:", len(rows))
    print("Plot:", plot_path)
    print("CSV:", csv_path)
    print("JSON:", json_path)
    print("Note: ResNet-50 U-Net substitute, not actual fine-tuned Prithvi.")


if __name__ == "__main__":
    main()
