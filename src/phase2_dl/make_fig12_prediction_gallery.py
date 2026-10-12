
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import segmentation_models_pytorch as smp

# GeoSense Phase 2 - Fig. 12 prediction gallery
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHIP_DIR = os.path.join(ROOT, "data", "satellite", "colab_small", "chips", "2023")
LABEL_FILE = os.path.join(ROOT, "data", "satellite", "labels", "labels_2023.npy")
UNET_FILE = os.path.join(ROOT, "models", "saved", "unet_best.pth")
SUBSTITUTE_FILE = os.path.join(ROOT, "models", "saved", "prithvi_finetuned.pth")
OUT_FILE = os.path.join(ROOT, "outputs", "plots", "fig12_heldout_prediction_gallery.png")

NUM_CLASSES = 5
SIZE = 224
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["Urban", "Vegetation", "Water", "Bare Land", "Agriculture"]
COLORS = ["#d73027", "#1a9850", "#4575b4", "#d9b66f", "#fee08b"]
cmap = ListedColormap(COLORS)
cmap.set_bad(color="white")

def load_model(path, encoder):
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=6,
        classes=NUM_CLASSES,
    )
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def read_chip(filename):
    path = os.path.join(CHIP_DIR, filename)
    with rasterio.open(path) as src:
        image = src.read(indexes=[1, 2, 3, 4, 5, 6]).astype(np.float32)

    # Match the evaluation preprocessing.
    for band in range(image.shape[0]):
        values = image[band]
        valid = np.isfinite(values)
        if valid.any():
            values[~valid] = np.nanmean(values)
        else:
            values[:] = 0

    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    image = np.clip(image, 0, 1)

    # Filename example: guwahati_2023_row_00000_col_00000.tif
    parts = os.path.splitext(filename)[0].split("_")
    row, col = int(parts[3]), int(parts[5])
    labels = np.load(LABEL_FILE, mmap_mode="r")
    label = np.array(labels[row:row + SIZE, col:col + SIZE])

    if label.shape != (SIZE, SIZE):
        raise ValueError(f"Unexpected label shape for {filename}: {label.shape}")

    return image, label

def show_label(array):
    valid = (array >= 0) & (array < NUM_CLASSES)
    result = np.ma.masked_where(~valid, array)
    return result

def main():
    for path in [LABEL_FILE, UNET_FILE, SUBSTITUTE_FILE]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    files = sorted(f for f in os.listdir(CHIP_DIR) if f.lower().endswith(".tif"))
    if len(files) < 5:
        raise RuntimeError(f"Expected at least 5 chips; found {len(files)}")

    # Reproduce the same held-out split as evaluate_models.py.
    rng = np.random.default_rng(SEED)
    indices = np.arange(len(files))
    rng.shuffle(indices)
    split = int(0.8 * len(indices))
    heldout = sorted(indices[split:])

    # Pick four samples spread across the held-out set.
    positions = np.linspace(0, len(heldout) - 1, 4, dtype=int)
    selected = [files[int(heldout[p])] for p in positions]

    unet = load_model(UNET_FILE, "resnet34")
    substitute = load_model(SUBSTITUTE_FILE, "resnet50")

    fig, axes = plt.subplots(4, 4, figsize=(15, 14))
    column_titles = [
        "True-colour satellite image",
        "Rule-based label",
        "U-Net prediction",
        "ResNet-50 U-Net substitute prediction",
    ]

    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight="bold", pad=12)

    with torch.no_grad():
        for row_idx, filename in enumerate(selected):
            image, label = read_chip(filename)
            tensor = torch.from_numpy(image).unsqueeze(0).to(device)

            pred_unet = torch.argmax(unet(tensor), dim=1)[0].cpu().numpy()
            pred_sub = torch.argmax(substitute(tensor), dim=1)[0].cpu().numpy()

            # The chip bands are B2, B3, B4, B8, B11, B12.
            # Display true colour as B4 red, B3 green, B2 blue.
            rgb = np.stack([image[2], image[1], image[0]], axis=-1)
            low = np.percentile(rgb, 2)
            high = np.percentile(rgb, 98)
            if high > low:
                rgb = np.clip((rgb - low) / (high - low), 0, 1)
            else:
                rgb = np.clip(rgb, 0, 1)

            panels = [
                ("image", rgb),
                ("label", show_label(label)),
                ("label", show_label(pred_unet)),
                ("label", show_label(pred_sub)),
            ]

            for col, (kind, data) in enumerate(panels):
                ax = axes[row_idx, col]
                if kind == "image":
                    ax.imshow(data)
                else:
                    ax.imshow(data, cmap=cmap, vmin=0, vmax=NUM_CLASSES - 1,
                              interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(
                        f"Chip {row_idx + 1}",
                        fontsize=10,
                        rotation=0,
                        labelpad=35,
                        va="center",
                    )

    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=9,
                   markerfacecolor=COLORS[i], markeredgecolor="none",
                   label=CLASS_NAMES[i])
        for i in range(NUM_CLASSES)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               frameon=False, bbox_to_anchor=(0.5, 0.015))
    fig.suptitle(
        "Guwahati Held-out Chip Predictions: Rule Labels and Model Outputs",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0.055, 1, 0.96], h_pad=1.0, w_pad=0.6)
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nFIG. 12 GALLERY GENERATED SUCCESSFULLY")
    print("Samples:", len(selected))
    print("Output:", OUT_FILE)
    print("Models: ResNet-34 U-Net and ResNet-50 U-Net substitute")
    print("Note: The second model is a U-Net substitute, not actual Prithvi weights.")

if __name__ == "__main__":
    main()
