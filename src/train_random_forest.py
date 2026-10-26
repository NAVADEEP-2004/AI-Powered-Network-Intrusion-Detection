
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score
)

# 1. Load the same training and testing data
X_train = joblib.load("artifacts/X_train.pkl")
X_test = joblib.load("artifacts/X_test.pkl")
y_train = joblib.load("artifacts/y_train.pkl")
y_test = joblib.load("artifacts/y_test.pkl")

print("Datasets loaded successfully!")
print("Training Random Forest...")

# 2. Create the model
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

# 3. Train
model.fit(X_train, y_train)

# 4. Predict
y_pred = model.predict(X_test)

# 5. Evaluate
print("\n=== RANDOM FOREST RESULTS ===")
print("Balanced accuracy:",
      balanced_accuracy_score(y_test, y_pred))

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["BENIGN", "BOT"],
    zero_division=0
))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

# 6. Save without overwriting Logistic Regression
joblib.dump(model, "artifacts/random_forest.pkl")
print("\nRandom Forest model saved successfully!")
