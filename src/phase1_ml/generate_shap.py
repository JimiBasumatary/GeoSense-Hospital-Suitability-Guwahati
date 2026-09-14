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
print("SHAP EXPLAINABILITY ANALYSIS")
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


X = df[FEATURES].copy()
y = df["label"].copy()


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

for column in X.columns:
    if X[column].isna().any():
        X[column] = X[column].fillna(X[column].median())

print("Number of model features:", X.shape[1])
print("Missing values in X:", X.isna().sum().sum())
print("Missing values in y:", y.isna().sum())


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

# Use the test data for explanation
X_explain = X_test.copy()

shap_values = explainer.shap_values(X_explain)


# ============================================================
# HANDLE DIFFERENT SHAP OUTPUT FORMATS
# ============================================================

if isinstance(shap_values, list):

    # Older SHAP format:
    # list containing one array for each class

    mean_abs_shap = np.mean(
        [
            np.abs(values).mean(axis=0)
            for values in shap_values
        ],
        axis=0
    )

else:

    shap_array = np.asarray(shap_values)

    if shap_array.ndim == 3:
        # Shape usually:
        # samples × features × classes

        mean_abs_shap = np.mean(
            np.abs(shap_array),
            axis=(0, 2)
        )

    elif shap_array.ndim == 2:
        mean_abs_shap = np.abs(shap_array).mean(axis=0)

    else:
        raise ValueError(
            f"Unexpected SHAP array shape: {shap_array.shape}"
        )


# ============================================================
# CREATE FEATURE IMPORTANCE TABLE
# ============================================================

importance_df = pd.DataFrame({
    "feature": FEATURES,
    "mean_absolute_shap": mean_abs_shap
})

importance_df = importance_df.sort_values(
    "mean_absolute_shap",
    ascending=False
)

print("\nSHAP FEATURE IMPORTANCE")
print("-" * 60)

for _, row in importance_df.iterrows():
    print(
        f"{row['feature']:<35} "
        f"{row['mean_absolute_shap']:.6f}"
    )


# ============================================================
# SAVE SHAP IMPORTANCE TABLE
# ============================================================

importance_file = os.path.join(
    OUTPUT_DIR,
    "shap_feature_importance.csv"
)

importance_df.to_csv(
    importance_file,
    index=False
)

print("\nFeature importance table saved to:")
print(importance_file)


# ============================================================
# CREATE SHAP BAR CHART
# ============================================================

plot_df = importance_df.sort_values(
    "mean_absolute_shap",
    ascending=True
)

plt.figure(figsize=(10, 8))

plt.barh(
    plot_df["feature"],
    plot_df["mean_absolute_shap"]
)

plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Feature")
plt.title(
    "SHAP Feature Importance for Hospital Suitability Prediction"
)

plt.tight_layout()


# ============================================================
# SAVE FIGURE
# ============================================================

figure_file = os.path.join(
    OUTPUT_DIR,
    "shap_importance.png"
)

plt.savefig(
    figure_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSHAP figure saved to:")
print(figure_file)


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("SHAP ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 60)