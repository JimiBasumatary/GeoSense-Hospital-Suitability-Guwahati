import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = r"D:\GeoSense_Agent"

DATA_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "labelled_sites.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "reports"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FEATURES USED FOR MACHINE LEARNING
# ============================================================

FEATURES = [
    "dist_road_m",
    "dist_hospital_m",
    "population_value",
    "elevation_m",
    "dist_landuse_commercial_m",
    "land_use_cemetery",
    "land_use_commercial",
    "land_use_farmland",
    "land_use_forest",
    "land_use_grass",
    "land_use_industrial",
    "land_use_meadow",
    "land_use_military",
    "land_use_orchard",
    "land_use_park",
    "land_use_quarry",
    "land_use_recreation_ground",
    "land_use_residential",
    "land_use_retail",
    "land_use_unclassified"
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("SHAP SUMMARY / BEESWARM ANALYSIS")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully.")
print("Number of records:", len(df))


# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [
    feature for feature in FEATURES
    if feature not in df.columns
]

if missing_features:
    print("\nERROR: Missing features:")

    for feature in missing_features:
        print(" -", feature)

    raise SystemExit(1)


# ============================================================
# PREPARE FEATURES AND LABEL
# ============================================================

X = df[FEATURES].copy()
y = df["label"].copy()


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

for column in X.columns:

    if X[column].isna().any():
        X[column] = X[column].fillna(
            X[column].median()
        )


print("\nNumber of model features:", X.shape[1])
print("Label classes:", sorted(y.unique()))
print("Number of classes:", y.nunique())


# ============================================================
# TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# TRAIN XGBOOST MODEL
# ============================================================

print("\nTraining XGBoost model...")

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42
)

model.fit(X_train, y_train)

print("XGBoost model trained successfully.")


# ============================================================
# CREATE SHAP EXPLAINER
# ============================================================

print("\nCreating SHAP explanations...")

explainer = shap.TreeExplainer(model)

X_explain = X_test.copy()

shap_values = explainer.shap_values(X_explain)


# ============================================================
# CONVERT SHAP VALUES FOR MULTICLASS MODEL
# ============================================================

if isinstance(shap_values, list):

    # Older SHAP format:
    # one array for each class

    shap_class = shap_values[2]

else:

    shap_array = np.asarray(shap_values)

    if shap_array.ndim == 3:

        # samples × features × classes
        shap_class = shap_array[:, :, 2]

    elif shap_array.ndim == 2:

        shap_class = shap_array

    else:

        raise ValueError(
            f"Unexpected SHAP array shape: {shap_array.shape}"
        )


print("SHAP values prepared successfully.")


# ============================================================
# CREATE SHAP SUMMARY / BEESWARM PLOT
# ============================================================

print("\nCreating SHAP summary plot...")

plt.figure(figsize=(10, 8))

shap.summary_plot(
    shap_class,
    X_explain,
    feature_names=FEATURES,
    show=False,
    max_display=20
)

plt.title(
    "SHAP Summary Plot for Hospital Suitability Prediction"
)

plt.tight_layout()


# ============================================================
# SAVE SHAP SUMMARY FIGURE
# ============================================================

figure_file = os.path.join(
    OUTPUT_DIR,
    "shap_summary.png"
)

plt.savefig(
    figure_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nSHAP summary figure saved to:")
print(figure_file)


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("SHAP SUMMARY ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 60)