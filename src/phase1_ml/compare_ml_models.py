import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


X_INPUT = r"data\processed\X_features.csv"
Y_INPUT = r"data\processed\y_target.csv"

RANDOM_STATE = 42


def compare_models():

    print("\n" + "=" * 75)
    print("GeoSense - Machine Learning Model Comparison")
    print("=" * 75)

    # ---------------------------------------------------------
    # 1. Read data
    # ---------------------------------------------------------
    print("\n1. Reading datasets...")

    X = pd.read_csv(X_INPUT)
    y = pd.read_csv(Y_INPUT).squeeze()

    # Remove flood-risk placeholder
    if "flood_risk" in X.columns:
        X = X.drop(columns=["flood_risk"])

    print("Features:", X.shape)
    print("Target  :", y.shape)

    # ---------------------------------------------------------
    # 2. Train-test split
    # ---------------------------------------------------------
    print("\n2. Creating stratified train-test split...")

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
    # 3. Define models
    # ---------------------------------------------------------
    models = {

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced"
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE
        )
    }

    results = []

    # ---------------------------------------------------------
    # 4. Train and evaluate
    # ---------------------------------------------------------
    for name, model in models.items():

        print("\n" + "-" * 75)
        print("Training:", name)
        print("-" * 75)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            average="macro"
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="macro"
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="macro"
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1
        })

        print("Accuracy :", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall   :", round(recall, 4))
        print("F1-score :", round(f1, 4))

    # ---------------------------------------------------------
    # 5. Comparison table
    # ---------------------------------------------------------
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 75)
    print("MODEL COMPARISON")
    print("=" * 75)

    print(
        results_df.round(4).to_string(index=False)
    )

    # ---------------------------------------------------------
    # 6. Select best model
    # ---------------------------------------------------------
    best_index = results_df["F1_Score"].idxmax()

    best_model = results_df.loc[
        best_index,
        "Model"
    ]

    print("\nBest model based on macro F1-score:")
    print(best_model)

    print("\n" + "=" * 75)
    print("MODEL COMPARISON COMPLETED")
    print("=" * 75)

    print("\nNext stage:")
    print("Train the selected model on the full dataset and")
    print("generate suitability predictions for all candidate locations.")

    print("=" * 75)


if __name__ == "__main__":
    compare_models()