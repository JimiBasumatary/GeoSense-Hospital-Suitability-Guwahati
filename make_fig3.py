
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# Project paths
root = r"D:\GeoSense_Agent"

label_2015_path = os.path.join(
    root, "data", "satellite", "labels", "labels_2015.npy"
)
label_2023_path = os.path.join(
    root, "data", "satellite", "labels", "labels_2023.npy"
)
output_path = os.path.join(
    root, "outputs", "fig3_guwahati_labels_2015_2023.png"
)

os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Load the actual label arrays
print("Loading Guwahati 2015 labels...")
labels_2015 = np.load(label_2015_path)

print("Loading Guwahati 2023 labels...")
labels_2023 = np.load(label_2023_path)

print("2015 dimensions:", labels_2015.shape)
print("2023 dimensions:", labels_2023.shape)

if labels_2015.shape != labels_2023.shape:
    raise ValueError("The two label arrays have different dimensions.")

# Class IDs:
# 0 Urban, 1 Vegetation, 2 Water,
# 3 Bare Land, 4 Agriculture, 255 NoData
colors = [
    "#d73027",  # Urban
    "#1a9850",  # Vegetation
    "#4575b4",  # Water
    "#bdbdbd",  # Bare Land
    "#fee08b"   # Agriculture
]

names = [
    "Urban", "Vegetation", "Water",
    "Bare Land", "Agriculture"
]

cmap = ListedColormap(colors)
cmap.set_bad("black")
norm = BoundaryNorm(np.arange(-0.5, 5.5, 1), cmap.N)

# Mask NoData and any unexpected class values
valid_2015 = np.isin(labels_2015, [0, 1, 2, 3, 4])
valid_2023 = np.isin(labels_2023, [0, 1, 2, 3, 4])

map_2015 = np.ma.masked_where(~valid_2015, labels_2015)
map_2023 = np.ma.masked_where(~valid_2023, labels_2023)

# Plot both years using the same colours and scale
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].imshow(map_2015, cmap=cmap, norm=norm,
               interpolation="nearest")
axes[0].set_title("Guwahati Pseudo-Labels (2015)")
axes[0].axis("off")

axes[1].imshow(map_2023, cmap=cmap, norm=norm,
               interpolation="nearest")
axes[1].set_title("Guwahati Pseudo-Labels (2023)")
axes[1].axis("off")

# Shared legend
handles = [
    plt.Line2D(
        [0], [0], marker="s", linestyle="",
        markerfacecolor=colors[i],
        markeredgecolor="black",
        markersize=9, label=names[i]
    )
    for i in range(5)
]

fig.legend(handles=handles, loc="lower center",
           ncol=5, frameon=False)

fig.suptitle(
    "Guwahati Rule-Based Pseudo-Label Maps: 2015 and 2023"
)

plt.tight_layout(rect=[0, 0.10, 1, 0.93])
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nFigure saved successfully:")
print(output_path)
