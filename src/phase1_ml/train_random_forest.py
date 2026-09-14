import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


X_INPUT = r"data\processed\X_features.csv"
Y_INPUT = r"data\processed\y_target.csv"

RANDOM_STATE = 42


def train_random_forest():

    print("\n" + "=" * 75)
    print("GeoSense - Random Forest Hospital Suitability Model")
    print("=" * 75)

    # ---------------------------------------------------------
    # 1. Read datasets
    # ---------------------------------------------------------
    print("\n1. Reading training datasets...")

    X = pd.read_csv(X_INPUT)
    y = pd.read_csv(Y_INPUT).squeeze()

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # ---------------------------------------------------------
    # 2. Remove flood-risk placeholder
    # ---------------------------------------------------------
    print("\n2. Preparing model features...")

    if "flood_risk" in X.columns:
        X = X.drop(columns=["flood_risk"])
        print("Removed flood_risk because it is a placeholder.")

    print("Final number of model features:", X.shape[1])

    # ---------------------------------------------------------
    # 3. Train-test split
    # ---------------------------------------------------------
    print("\n3. Creating train-test split...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("Training samples:", len(X_train))
    print("Testing samples :", len(X_test))

    # ---------------------------------------------------------
    # 4. Create Random Forest model
    # ---------------------------------------------------------
    print("\n4. Creating Random Forest classifier...")

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )

    # ---------------------------------------------------------
    # 5. Train model
    # ---------------------------------------------------------
    print("\n5. Training Random Forest...")

    model.fit(X_train, y_train)

    print("Model training completed.")

    # ---------------------------------------------------------
    # 6. Predictions
    # ---------------------------------------------------------
    print("\n6. Generating predictions...")

    y_pred = model.predict(X_test)

    # ---------------------------------------------------------
    # 7. Accuracy
    # ---------------------------------------------------------
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "-" * 75)
    print("7. Model Accuracy")
    print("-" * 75)

    print(
        "Accuracy:",
        round(accuracy, 4)
    )

    print(
        "Accuracy (%):",
        round(accuracy * 100, 2)
    )

    # ---------------------------------------------------------
    # 8. Classification report
    # ---------------------------------------------------------
    print("\n" + "-" * 75)
    print("8. Classification Report")
    print("-" * 75)

    print(
        classification_report(
            y_test,
            y_pred,
            digits=4
        )
    )

    # ---------------------------------------------------------
    # 9. Confusion matrix
    # ---------------------------------------------------------
    print("\n" + "-" * 75)
    print("9. Confusion Matrix")
    print("-" * 75)

    labels = ["Low", "Moderate", "High"]

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    print(
        pd.DataFrame(
            cm,
            index=[
                "Actual Low",
                "Actual Moderate",
                "Actual High"
            ],
            columns=[
                "Predicted Low",
                "Predicted Moderate",
                "Predicted High"
            ]
        )
    )

    # ---------------------------------------------------------
    # 10. Feature importance
    # ---------------------------------------------------------
    print("\n" + "-" * 75)
    print("10. Feature Importance")
    print("-" * 75)

    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    print(
        importance.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # 11. Final information
    # ---------------------------------------------------------
    print("\n" + "=" * 75)
    print("RANDOM FOREST TRAINING COMPLETED")
    print("=" * 75)

    print("\nModel:")
    print("Random Forest Classifier")

    print("Number of trees:", 300)

    print("Random state:", RANDOM_STATE)

    print("\nFinal model features:", X.shape[1])

    print("\nNext stage:")
    print("Compare the Random Forest results with another ML model.")

    print("=" * 75)


if __name__ == "__main__":
    train_random_forest()