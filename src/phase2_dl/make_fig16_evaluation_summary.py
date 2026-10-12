
import os
import json
import matplotlib.pyplot as plt

ROOT = r"D:\GeoSense_Agent"
RESULTS = os.path.join(
    ROOT, "outputs", "plots", "model_evaluation_results.json"
)
OUTPUT = os.path.join(
    ROOT, "outputs", "plots", "fig16_evaluation_summary.png"
)

with open(RESULTS, "r", encoding="utf-8") as f:
    data = json.load(f)

unet = data["unet"]
sub = data["foundation_model"]

fig = plt.figure(figsize=(13, 8))
fig.suptitle(
    "GeoSense Agent — Guwahati 2023 Evaluation Summary",
    fontsize=17, fontweight="bold", y=0.97
)

ax = fig.add_axes([0.06, 0.08, 0.88, 0.80])
ax.axis("off")

summary = [
    ("STUDY AREA", "Guwahati, Assam | 2023 satellite data"),
    ("EVALUATION", f'{data["total_chips"]} chips; '
                   f'{data["heldout_evaluation_chips"]} held-out chips'),
    ("LABELLED PIXELS", f'{data["labelled_pixels_evaluated"]:,} '
                        "valid pixels; NoData 255 excluded"),
    ("U-NET ACCURACY", f'{unet["accuracy_percent"]:.2f}%'),
    ("RESNET-50 U-NET SUBSTITUTE ACCURACY",
     f'{sub["accuracy_percent"]:.2f}%'),
    ("U-NET MEAN IoU", f'{unet["mean_iou"] * 100:.2f}%'),
    ("SUBSTITUTE MEAN IoU", f'{sub["mean_iou"] * 100:.2f}%'),
    ("U-NET MACRO F1", f'{unet["macro_f1"] * 100:.2f}%'),
    ("SUBSTITUTE MACRO F1", f'{sub["macro_f1"] * 100:.2f}%'),
    ("OUTPUTS", "Confusion matrices, evaluation JSON, "
                "model comparison report, training curves, "
                "per-class IoU and per-chip accuracy plots"),
]

y = 0.92
for label, value in summary:
    ax.text(
        0.02, y, label, transform=ax.transAxes,
        fontsize=10, fontweight="bold", va="top"
    )
    ax.text(
        0.40, y, value, transform=ax.transAxes,
        fontsize=10, va="top", wrap=True
    )
    y -= 0.085

ax.text(
    0.02, 0.035,
    "Evaluation uses rule-based pseudo-labels, not independently "
    "verified ground truth. The ResNet-50 model is a U-Net substitute, "
    "not actual fine-tuned Prithvi-100M weights. Runtime and tile count "
    "were not included because this summary does not measure them.",
    transform=ax.transAxes, fontsize=8, wrap=True
)

fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
plt.close(fig)

print("FIG. 16 SUMMARY GENERATED")
print("Output:", OUTPUT)
print("U-Net accuracy:", f'{unet["accuracy_percent"]:.2f}%')
print("Substitute accuracy:", f'{sub["accuracy_percent"]:.2f}%')
print("Held-out chips:", data["heldout_evaluation_chips"])
print("Labelled pixels:", f'{data["labelled_pixels_evaluated"]:,}')
print("Runtime: not measured by this summary script")
print("Tile count: not reported by this summary script")
