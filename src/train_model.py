
# import joblib
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import (
#     classification_report,
#     confusion_matrix,
#     balanced_accuracy_score
# )

# # Step 1: Load the saved datasets
# X_train = joblib.load("artifacts/X_train.pkl")
# X_test = joblib.load("artifacts/X_test.pkl")
# y_train = joblib.load("artifacts/y_train.pkl")
# y_test = joblib.load("artifacts/y_test.pkl")

# print("Datasets loaded successfully!")
# print("Training data shape:", X_train.shape)
# print("Testing data shape:", X_test.shape)

# # Step 2: Create the baseline model
# model = LogisticRegression(
#     max_iter=1000,
#     class_weight="balanced",
#     random_state=42
# )

# # Step 3: Train the model
# print("\nTraining Logistic Regression...")
# model.fit(X_train, y_train)

# # Step 4: Make predictions
# y_pred = model.predict(X_test)

# # Step 5: Evaluate the model
# print("\n=== BASELINE MODEL RESULTS ===")
# print("Balanced accuracy:", balanced_accuracy_score(y_test, y_pred))

# print("\nClassification report:")
# print(classification_report(
#     y_test,
#     y_pred,
#     target_names=["BENIGN", "BOT"],
#     zero_division=0
# ))

# print("\nConfusion matrix:")
# print(confusion_matrix(y_test, y_pred))

# # Step 6: Save the trained model
# joblib.dump(model, "artifacts/logistic_regression.pkl")
# print("\nModel saved to artifacts/logistic_regression.pkl")


import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score
)

# 1. Load the saved datasets
X_train = joblib.load("artifacts/X_train.pkl")
X_test = joblib.load("artifacts/X_test.pkl")
y_train = joblib.load("artifacts/y_train.pkl")
y_test = joblib.load("artifacts/y_test.pkl")

print("Datasets loaded successfully!")

# 2. Create a pipeline: scaling + Logistic Regression
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

# 3. Train
print("Training scaled Logistic Regression...")
model.fit(X_train, y_train)

# 4. Predict
y_pred = model.predict(X_test)

# 5. Evaluate
print("\n=== SCALED LOGISTIC REGRESSION ===")
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

# 6. Save the model
joblib.dump(model, "artifacts/logistic_regression_scaled.pkl")
print("\nScaled model saved successfully!")
