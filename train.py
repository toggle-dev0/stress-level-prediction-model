from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Tuning
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler


def main():
    data = pd.read_csv("ScreenTime_with_Stress_Classes.csv")
    dumped_features = [
        "user_id",
        "age",
        "gender",
        "occupation",
        "work_mode",
        "screen_time_hours",
        "social_hours_per_week",
        "productivity_0_100",
        "stress_level_0_10",
        "mental_wellness_index_0_100",
    ]
    data = data.drop(dumped_features, axis=1)
    X = data.drop("stress_class", axis=1)
    y = data["stress_class"]

    # Encode stress_class labels to numerical values
    label_encoding = {"Low": 0, "Medium": 1, "High": 2}
    data["stress_class"] = data["stress_class"].map(label_encoding)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    smote = SMOTE(k_neighbors=3, random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    model = LogisticRegression(random_state=42)
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test_scaled)
    result = classification_report(y_test, y_pred, zero_division=0)

    # Fine-tune model using GridSearchCV
    # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # param_grid = [
    #     # l2 penalty (default, most common)
    #     {
    #         "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
    #         "model__penalty": ["l2"],
    #         "model__solver": ["saga"],
    #         "model__max_iter": [2000],
    #     },
    #     # l1 penalty
    #     {
    #         "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
    #         "model__penalty": ["l1"],
    #         "model__solver": ["saga"],
    #         "model__max_iter": [2000],
    #     },
    #     # elasticnet penalty
    #     {
    #         "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
    #         "model__penalty": ["elasticnet"],
    #         "model__solver": ["saga"],
    #         "model__l1_ratio": [0.25, 0.5, 0.75],
    #         "model__max_iter": [2000],
    #     },
    # ]

    # pipe = ImbPipeline(
    #     [
    #         ("scaler", StandardScaler()),
    #         ("smote", SMOTE(k_neighbors=3, random_state=42)),
    #         ("model", LogisticRegression(class_weight="balanced", random_state=42)),
    #     ]
    # )

    # grid_search = GridSearchCV(
    #     pipe, param_grid, cv=cv, scoring="balanced_accuracy", n_jobs=-1, verbose=1
    # )

    # grid_search.fit(X, y)

    # print(f"Best BalAcc: {grid_search.best_score_:.3f}")
    # print(f"Best Params: {grid_search.best_params_}")

    # Final Model Parameters
    final_model = ImbPipeline(
        [
            ("scaler", StandardScaler()),
            ("smote", SMOTE(k_neighbors=5, random_state=42)),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    C=0.1,
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=0.5,
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    # Train on full training data and evaluate on test set
    final_model.fit(X_train_balanced, y_train_balanced)
    y_pred = final_model.predict(X_test_scaled)

    # Classification Report
    # print("Classification Report:")
    result = classification_report(
        y_test, y_pred, target_names=["Low", "Medium", "High"], zero_division=0
    )

    accuracy = accuracy_score(y_test, y_pred)

    # Confusion Matrix
    # cm = confusion_matrix(y_test, y_pred)

    # plt.figure(figsize=(6, 5))
    # sns.heatmap(
    #     cm,
    #     annot=True,
    #     fmt="d",
    #     cmap="Blues",
    #     xticklabels=["Low", "Medium", "High"],
    #     yticklabels=["Low", "Medium", "High"],
    # )
    # plt.title("Confusion Matrix — Tuned Logistic Regression")
    # plt.ylabel("Actual")
    # plt.xlabel("Predicted")
    # plt.tight_layout()
    # plt.show()

    artifact = {
        "model": final_model,
        "target_names": ["Low", "Medium", "High"],  # your stress classes
        "feature_names": list(X.columns),  # your df column names
    }

    output_path = Path("artifacts/stress_level_prediction_model.joblib")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    print(f"Model saved to: {output_path}")
    print(f"Test Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
