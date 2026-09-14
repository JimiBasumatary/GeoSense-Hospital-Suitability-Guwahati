import os
import joblib
import pandas as pd

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

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "saved"
)

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "hospital_suitability_xgboost.joblib"
)


# ============================================================
# FEATURES
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
print("SAVING FINAL XGBOOST MODEL")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully.")
print("Number of records:", len(df))


# ============================================================
# PREPARE FEATURES AND TARGET
# ============================================================

X = df[FEATURES].copy()
y = df["label"].copy()


# Fill missing predictor values using training-dataset medians
feature_medians = X.median()

X = X.fillna(feature_medians)


print("Number of features:", X.shape[1])
print("Missing values after imputation:", X.isna().sum().sum())


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
# TRAIN FINAL XGBOOST MODEL
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
# SAVE MODEL INFORMATION
# ============================================================

saved_model = {
    "model": model,
    "features": FEATURES,
    "feature_medians": feature_medians.to_dict(),
    "class_labels": {
        0: "Low",
        1: "Moderate",
        2: "High"
    }
}


joblib.dump(
    saved_model,
    MODEL_FILE
)


# ============================================================
# VERIFY FILE
# ============================================================

print("\nFinal model saved to:")
print(MODEL_FILE)

print("\nSaved feature count:", len(FEATURES))

print("\nClass mapping:")
print("0 = Low")
print("1 = Moderate")
print("2 = High")

print("\n" + "=" * 60)
print("FINAL XGBOOST MODEL SAVED SUCCESSFULLY")
print("=" * 60)