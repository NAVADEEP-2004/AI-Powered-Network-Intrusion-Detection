
import joblib

# Load the trained Random Forest model
model = joblib.load("artifacts/random_forest.pkl")

# Load the training features
X_train = joblib.load("artifacts/X_train.pkl")

print("=== MODEL FEATURE CHECK ===")

print("Number of training features:", X_train.shape[1])
print("Training data type:", type(X_train).__name__)

if hasattr(X_train, "columns"):
    print("\nFeature names:")
    print(X_train.columns.tolist())
else:
    print("\nTraining data has no column names.")

if hasattr(model, "n_features_in_"):
    print("\nFeatures expected by model:", model.n_features_in_)

if hasattr(model, "feature_names_in_"):
    print("\nModel's recorded feature names:")
    print(model.feature_names_in_.tolist())
else:
    print("\nModel does not store feature names.")

print("\nFeature check completed.")
