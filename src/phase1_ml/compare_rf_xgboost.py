import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


print("=" * 70)
print("GeoSense - Random Forest and XGBoost Model Comparison")
print("Hospital Site Suitability - Guwahati City")
print("=" * 70)


# ---------------------------------------------------------
# 1. Define project paths
# ---------------------------------------------------------

project_root = Path(__file__).resolve().parents[2]

input_file = (
    project_root
    / "data"
    / "processed"
    / "labelled_sites.csv"
)


# ---------------------------------------------------------
# 2. Load labelled dataset
# ---------------------------------------------------------

df = pd.read_csv(input_file)

print("\nDataset loaded successfully.")
print("Number of records:", len(df))


# ---------------------------------------------------------
# 3. Define model features
# ---------------------------------------------------------

features = [
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

target = "label"


print("\nNumber of model features:", len(features))

print("\nModel features:")
for feature in features:
    print(" -", feature)


# ---------------------------------------------------------
# 4. Prepare X and y
# ---------------------------------------------------------

X = df[features].copy()
y = df[target].copy()


# ---------------------------------------------------------
# 5. Handle missing numerical values
# ---------------------------------------------------------

for column in X.columns:
    if X[column].isna().any():
        X[column] = X[column].fillna(X[column].median())


print("\nMissing values in X:", X.isna().sum().sum())
print("Missing values in y:", y.isna().sum())


# ---------------------------------------------------------
# 6. Train-test split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nStratified train-test split:")
print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ---------------------------------------------------------
# 7. Train Random Forest
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)


# ---------------------------------------------------------
# 8. Evaluate Random Forest
# ---------------------------------------------------------

rf_accuracy = accuracy_score(y_test, rf_predictions)

rf_precision = precision_score(
    y_test,
    rf_predictions,
    average="macro"
)

rf_recall = recall_score(
    y_test,
    rf_predictions,
    average="macro"
)

rf_f1 = f1_score(
    y_test,
    rf_predictions,
    average="macro"
)

print("\nRandom Forest Performance")
print("-------------------------")
print("Accuracy :", round(rf_accuracy, 4))
print("Precision:", round(rf_precision, 4))
print("Recall   :", round(rf_recall, 4))
print("Macro F1 :", round(rf_f1, 4))

print("\nRandom Forest Classification Report:")
print(
    classification_report(
        y_test,
        rf_predictions,
        target_names=["Low", "Moderate", "High"]
    )
)

print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))


# ---------------------------------------------------------
# 9. Train XGBoost
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING XGBOOST")
print("=" * 70)

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

xgb_predictions = xgb_model.predict(X_test)


# ---------------------------------------------------------
# 10. Evaluate XGBoost
# ---------------------------------------------------------

xgb_accuracy = accuracy_score(y_test, xgb_predictions)

xgb_precision = precision_score(
    y_test,
    xgb_predictions,
    average="macro"
)

xgb_recall = recall_score(
    y_test,
    xgb_predictions,
    average="macro"
)

xgb_f1 = f1_score(
    y_test,
    xgb_predictions,
    average="macro"
)

print("\nXGBoost Performance")
print("-------------------")
print("Accuracy :", round(xgb_accuracy, 4))
print("Precision:", round(xgb_precision, 4))
print("Recall   :", round(xgb_recall, 4))
print("Macro F1 :", round(xgb_f1, 4))

print("\nXGBoost Classification Report:")
print(
    classification_report(
        y_test,
        xgb_predictions,
        target_names=["Low", "Moderate", "High"]
    )
)

print("XGBoost Confusion Matrix:")
print(confusion_matrix(y_test, xgb_predictions))


# ---------------------------------------------------------
# 11. Model comparison
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print("\nModel              Accuracy    Precision    Recall    Macro F1")
print("-" * 70)

print(
    f"Random Forest      "
    f"{rf_accuracy:.4f}       "
    f"{rf_precision:.4f}       "
    f"{rf_recall:.4f}     "
    f"{rf_f1:.4f}"
)

print(
    f"XGBoost            "
    f"{xgb_accuracy:.4f}       "
    f"{xgb_precision:.4f}       "
    f"{xgb_recall:.4f}     "
    f"{xgb_f1:.4f}"
)


# ---------------------------------------------------------
# 12. Select best model using Macro F1
# ---------------------------------------------------------

if rf_f1 >= xgb_f1:
    best_model = "Random Forest"
    best_f1 = rf_f1
else:
    best_model = "XGBoost"
    best_f1 = xgb_f1


print("\nSelected model:", best_model)
print("Best Macro F1:", round(best_f1, 4))


print("\n" + "=" * 70)
print("MODEL COMPARISON COMPLETED SUCCESSFULLY")
print("=" * 70)