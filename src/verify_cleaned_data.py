import pandas as pd
import numpy as np

# 1. Load the cleaned dataset
file_path = "data/CIC-IDS2017/Friday-Morning-cleaned.csv"
df = pd.read_csv(file_path)

# 2. Remove accidental spaces from column names
df.columns = df.columns.str.strip()

print("=== CLEANED DATASET VERIFICATION ===")

# 3. Check dataset size
print("\n1. Dataset shape:")
print(df.shape)

# 4. Check whether the target label exists
print("\n2. Label column:")
print("Label" in df.columns)

if "Label" not in df.columns:
    raise ValueError("Label column not found. Check the column names.")

# 5. Check class distribution
print("\n3. Label distribution:")
print(df["Label"].value_counts())

# 6. Check missing values
print("\n4. Total missing values:")
print(df.isnull().sum().sum())

# 7. Check infinite values in numeric columns
numeric_df = df.select_dtypes(include=np.number)
print("\n5. Total infinite values:")
print(np.isinf(numeric_df).sum().sum())

# 8. Check duplicate rows
print("\n6. Duplicate rows:")
print(df.duplicated().sum())

# 9. Check label names
print("\n7. Unique labels:")
print(df["Label"].unique())

print("\nVerification complete!")