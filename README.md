# Phase 2 – Deep Learning + Satellite Imagery

## Phase 2 Overview

Phase 2 extends the GeoSense hospital site-suitability project by integrating
satellite imagery, deep learning, semantic segmentation, and automated image
classification for the Guwahati study area.

The Phase 2 workflow includes:

1. Sentinel-2 satellite data acquisition for 2015 and 2023
2. Satellite image preprocessing
3. NDVI and NDWI generation
4. Image chip generation
5. Pseudo-label generation
6. U-Net semantic segmentation
7. Foundation-model experiment
8. Model evaluation using IoU and confusion matrices
9. Image classification using `image_classifier.py`
10. NDVI-based change detection

---

## Study Area

**Study Area:** Guwahati, Assam, India

The same Guwahati study area used in Phase 1 was retained for Phase 2.

---

## Satellite Data

Sentinel-2 imagery was used for two time periods:

- **2015**
- **2023**

The six input bands used for the deep-learning workflow were:

- B2 – Blue
- B3 – Green
- B4 – Red
- B8 – Near Infrared (NIR)
- B11 – SWIR 1
- B12 – SWIR 2

The processed imagery additionally contains:

- NDVI
- NDWI

---

## Image Preprocessing

The satellite imagery was processed using Python and Rasterio.

The preprocessing workflow included:

- Cloud/no-data masking
- Reflectance normalization
- Value clipping to the range 0–1
- NDVI calculation
- NDWI calculation
- GeoTIFF generation

### NDVI

NDVI was calculated using:

**NDVI = (NIR − Red) / (NIR + Red)**

### NDWI

NDWI was calculated using:

**NDWI = (Green − NIR) / (Green + NIR)**

---

## Phase 2 Data Statistics

| Year | Mean NDVI | Mean NDWI | Valid / Cloud-Free |
|------|-----------|-----------|---------------------|
| 2015 | 0.3385 | -0.3058 | 87.18% |
| 2023 | 0.4522 | -0.4169 | 89.58% |

---

## Image Chip Generation

The processed satellite imagery was divided into image chips for deep-learning training.

**Chip size:** 224 × 224 pixels

**Stride:** 174 pixels

**Overlap:** 50 pixels

| Year | Number of Chips |
|------|-----------------|
| 2015 | 501 |
| 2023 | 544 |

Remaining missing values in usable chips were filled using band-level mean values.

---

## Ground-Truth / Pseudo-Labels

Five land-cover classes were generated using spectral thresholds:

1. Urban
2. Vegetation
3. Water
4. Bare Land
5. Agriculture

The labels were generated automatically from NDVI, NDWI and SWIR information.

These are **pseudo-labels**, not manually digitized field-verified ground truth. Therefore, the model results should be interpreted within the limitations of the rule-based labelling approach.

---

## U-Net Semantic Segmentation

A U-Net architecture with a ResNet-34 encoder was implemented for semantic segmentation.

### Configuration

- Input bands: 6
- Number of classes: 5
- Encoder: ResNet-34
- Loss function: Cross Entropy Loss
- Optimizer: Adam
- Learning rate: 0.001
- Training/validation split: 80/20
- Data augmentation:
  - Horizontal flip
  - Vertical flip
  - 90° rotation
  - Brightness/contrast augmentation

### Training Result

The final validation accuracy obtained during training was:

**79.51%**

The trained models were saved as:

- `models/saved/unet_best.pth`
- `models/saved/unet_final.pth`

Training history and plots were also generated.

---

## Foundation Model Experiment

The Phase 2 workflow also investigated the use of the **Prithvi-100M** Earth-observation foundation model.

The Prithvi checkpoint was successfully accessed through Hugging Face. However, the checkpoint was not directly compatible with the implemented six-band U-Net segmentation architecture.

Therefore, a documented **ResNet-50 segmentation substitute** was used for the foundation-model experiment.

The substitute was trained without ImageNet pretrained weights because the required pretrained-weight download was unavailable.

### Model Parameters

- Total parameters: 32,531,093
- Trainable parameters: 10,449,045
- Frozen parameters: 22,082,048
- Backbone learning rate: 0.00001
- Head learning rate: 0.0001

The resulting model was saved as:

`models/saved/prithvi_finetuned.pth`

---

## Model Evaluation

The models were evaluated using the same held-out evaluation subset of 16 image chips.

### Overall Results

| Model | Accuracy | Mean IoU |
|-------|----------|----------|
| U-Net | 73.75% | 0.3010 |
| Foundation-model experiment | 52.33% | 0.1235 |

### U-Net Per-Class IoU

| Class | IoU |
|-------|-----|
| Urban | 0.0000 |
| Vegetation | 0.7932 |
| Water | 0.0004 |
| Bare Land | 0.2131 |
| Agriculture | 0.4985 |

### Foundation-Model Experiment Per-Class IoU

| Class | IoU |
|-------|-----|
| Urban | 0.0039 |
| Vegetation | 0.5786 |
| Water | 0.0014 |
| Bare Land | 0.0028 |
| Agriculture | 0.0307 |

The evaluation subset was the same held-out validation subset used during training/model selection; therefore, these results should not be interpreted as an independent test-set benchmark.

---

## Confusion Matrices

The following confusion-matrix plots were generated:

- `outputs/plots/unet_confusion_matrix.png`
- `outputs/plots/foundation_confusion_matrix.png`

A training-curve comparison was also generated:

- `outputs/plots/training_curve_comparison.png`

---

## Image Classification Module

The Phase 2 project includes:

`src/phase2_dl/image_classifier.py`

The module performs automated satellite-image classification for a supplied latitude and longitude.

The classifier returns:

- Land-cover class
- Class ID
- Confidence percentage
- NDVI
- NDWI
- NDVI interpretation
- Class distribution
- Change flag
- Change description
- 2015 NDVI
- 2023 NDVI
- NDVI difference
- Model used
- Inference time

---

## Change Detection

NDVI-based change detection was implemented by comparing corresponding 2015 and 2023 image patches.

A change is flagged when:

**Absolute NDVI difference > 0.15**

### Unchanged Example

Location:

**26.1445, 91.7362**

2015 NDVI: **0.1744**

2023 NDVI: **0.2584**

NDVI difference: **0.0841**

Change flag:

**False**

---

### Changed Example

Location:

**26.231480, 91.896261**

2015 NDVI: **0.2674**

2023 NDVI: **-0.0042**

NDVI difference: **0.2716**

Change flag:

**True**

The inference time for the changed-location test was approximately **1.41 seconds**, which is below the 3-second practical requirement.

---

## Phase 2 Project Structure

```text
GeoSense_Agent/
│
├── data/
│   ├── satellite/
│   │   ├── raw/
│   │   ├── processed/
│   │   ├── chips/
│   │   └── labels/
│   │
│   └── outputs/
│
├── models/
│   ├── saved/
│   └── evaluation/
│
├── src/
│   └── phase2_dl/
│       ├── download_sentinel2.py
│       ├── preprocess_imagery.py
│       ├── chip_images.py
│       ├── generate_labels.py
│       ├── train_unet.py
│       ├── finetune_prithvi.py
│       ├── evaluate_models.py
│       ├── compare_training_curves.py
│       └── image_classifier.py
│
└── README.md