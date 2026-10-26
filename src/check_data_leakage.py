
import joblib
import pandas as pd

# Load training and testing features
X_train = joblib.load("artifacts/X_train.pkl")
X_test = joblib.load("artifacts/X_test.pkl")

# Load labels
y_train = joblib.load("artifacts/y_train.pkl")
y_test = joblib.load("artifacts/y_test.pkl")

print("=== DATA LEAKAGE CHECK ===")

print("\n1. Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# Check whether identical feature rows occur in both sets
train_rows = set(
    pd.util.hash_pandas_object(X_train, index=False)
)
test_rows = set(
    pd.util.hash_pandas_object(X_test, index=False)
)

overlap = train_rows.intersection(test_rows)

print("\n2. Number of distinct feature-row hashes in both sets:")
print(len(overlap))

# Check class proportions
print("\n3. Training label proportions:")
print(y_train.value_counts(normalize=True).round(4))

print("\n4. Testing label proportions:")
print(y_test.value_counts(normalize=True).round(4))

print("\nCheck completed.")
