
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the labeled dataset
file_path = "data/CIC-IDS2017/Friday-Morning-labeled.csv"
df = pd.read_csv(file_path)

# Remove accidental spaces from column names
df.columns = df.columns.str.strip()

# Separate features and target
X = df.drop(columns=["Label", "target"])
y = df["target"]

# Split into 80% training and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("=== DATA SPLIT RESULTS ===")

print("\nTraining features:", X_train.shape)
print("Testing features:", X_test.shape)

print("\nTraining target:", y_train.shape)
print("Testing target:", y_test.shape)

print("\nTraining label distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting label distribution:")
print(y_test.value_counts().sort_index())

print("\nTraining attack percentage:",
      round(y_train.mean() * 100, 2), "%")

print("Testing attack percentage:",
      round(y_test.mean() * 100, 2), "%")

import joblib
import os

# Create a folder to store the split datasets
os.makedirs("artifacts", exist_ok=True)

# Save the training and testing data
joblib.dump(X_train, "artifacts/X_train.pkl")
joblib.dump(X_test, "artifacts/X_test.pkl")
joblib.dump(y_train, "artifacts/y_train.pkl")
joblib.dump(y_test, "artifacts/y_test.pkl")

print("Training and testing data saved successfully!")

print("\nData splitting completed successfully!")
