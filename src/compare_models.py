
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Step 1: Load test data
X_test = joblib.load("artifacts/X_test.pkl")
y_test = joblib.load("artifacts/y_test.pkl")

# Step 2: Load trained models
models = {
    "Logistic Regression": joblib.load(
        "artifacts/logistic_regression_scaled.pkl"
    ),
    "Random Forest": joblib.load(
        "artifacts/random_forest.pkl"
    )
}

# Step 3: Evaluate each model
results = []

for name, model in models.items():
    y_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(
            y_test, y_pred
        ),
        "Attack Precision": precision_score(
            y_test, y_pred, pos_label=1, zero_division=0
        ),
        "Attack Recall": recall_score(
            y_test, y_pred, pos_label=1, zero_division=0
        ),
        "Attack F1-score": f1_score(
            y_test, y_pred, pos_label=1, zero_division=0
        )
    })

# Step 4: Display and save comparison
comparison = pd.DataFrame(results)

print("\n=== MODEL COMPARISON ===")
print(comparison.round(4).to_string(index=False))

comparison.to_csv(
    "artifacts/model_comparison.csv", index=False
)

print("\nComparison saved to artifacts/model_comparison.csv")
