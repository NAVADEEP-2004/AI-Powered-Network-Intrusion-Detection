
import pandas as pd
import numpy as np

# Load the dataset with binary target labels
file_path = "data/CIC-IDS2017/Friday-Morning-labeled.csv"
df = pd.read_csv(file_path)

# Remove accidental spaces from column names
df.columns = df.columns.str.strip()

# Separate the target from the input features
y = df["target"]
X = df.drop(columns=["Label", "target"])

# Keep only numeric input features
X = X.select_dtypes(include=np.number)

print("=== FEATURE AND TARGET PREPARATION ===")

print("\nFeature matrix X shape:")
print(X.shape)

print("\nTarget vector y shape:")
print(y.shape)

print("\nFirst 5 feature rows:")
print(X.head())

print("\nTarget distribution:")
print(y.value_counts().sort_index())

print("\nNon-numeric feature columns remaining:")
print(X.select_dtypes(exclude=np.number).columns.tolist())

print("\nMissing values in X:")
print(X.isnull().sum().sum())

print("\nInfinite values in X:")
print(np.isinf(X).sum().sum())

print("\nFeatures and target prepared successfully!")
