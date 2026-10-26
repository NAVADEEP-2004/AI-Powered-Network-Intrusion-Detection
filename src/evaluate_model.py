
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# Load the saved test dataset
X_test = joblib.load("artifacts/X_test.pkl")
y_test = joblib.load("artifacts/y_test.pkl")

# Load the trained Random Forest model
model = joblib.load("artifacts/random_forest.pkl")

# Generate predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
results = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
    "Attack Precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
    "Attack Recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
    "Attack F1-score": f1_score(y_test, y_pred, pos_label=1, zero_division=0)
}

print("=== RANDOM FOREST EVALUATION ===")

for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["BENIGN", "BOT"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

# Save metrics for future use
pd.DataFrame([results]).to_csv(
    "artifacts/final_evaluation.csv",
    index=False
)

print("\nEvaluation saved to artifacts/final_evaluation.csv")
