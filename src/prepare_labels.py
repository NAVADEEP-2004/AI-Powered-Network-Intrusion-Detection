
import pandas as pd

# Load the cleaned dataset
file_path = "data/CIC-IDS2017/Friday-Morning-cleaned.csv"
df = pd.read_csv(file_path)

# Remove accidental spaces from column names
df.columns = df.columns.str.strip()

# Convert text labels into binary values
label_mapping = {
    "BENIGN": 0,
    "Bot": 1
}

df["target"] = df["Label"].map(label_mapping)

# Verify that every label was mapped successfully
if df["target"].isnull().any():
    raise ValueError("Some labels were not mapped. Check the original labels.")

df["target"] = df["target"].astype(int)

print("Original labels:")
print(df["Label"].value_counts())

print("\nBinary target distribution:")
print(df["target"].value_counts().sort_index())

print("\nTarget mapping:")
print("0 = BENIGN (normal traffic)")
print("1 = Bot (attack traffic)")

# Save the dataset with the new target column
output_path = "data/CIC-IDS2017/Friday-Morning-labeled.csv"
df.to_csv(output_path, index=False)

print("\nLabeled dataset saved successfully!")
print("File:", output_path)
print("Dataset shape:", df.shape)
