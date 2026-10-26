
import pandas as pd
import numpy as np

# Step 1: Load the original dataset
file_path = "data/CIC-IDS2017/Friday-WorkingHours-Morning.pcap_ISCX.csv"
df = pd.read_csv(file_path)

# Step 2: Clean column names
df.columns = df.columns.str.strip()

# Step 3: Replace positive and negative infinity with missing values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Step 4: Display the results
print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nTotal missing values after replacing infinity:")
print(df.isnull().sum().sum())

print("\nMissing values by column:")
print(df.isnull().sum()[df.isnull().sum() > 0])


# Step 5: Remove rows containing missing values
before_rows = len(df)

df_clean = df.dropna().copy()

after_rows = len(df_clean)

print("\n--- Missing-value cleaning results ---")
print("Rows before cleaning:", before_rows)
print("Rows after cleaning:", after_rows)
print("Rows removed:", before_rows - after_rows)

print("\nRemaining missing values:", df_clean.isnull().sum().sum())


# Step 6: Check duplicate rows after missing-value cleaning
duplicate_count = df_clean.duplicated().sum()

print("\n--- Duplicate analysis ---")
print("Duplicate rows:", duplicate_count)

# Count rows that belong to duplicate groups
duplicate_rows = df_clean[df_clean.duplicated(keep=False)]

print("\nLabels among duplicate groups:")
print(duplicate_rows["Label"].value_counts())

# Show the label distribution before removing duplicates
print("\nLabel distribution before duplicate removal:")
print(df_clean["Label"].value_counts())


# Step 7: Remove exact duplicate rows
before_dedup = len(df_clean)

df_clean = df_clean.drop_duplicates().copy()

after_dedup = len(df_clean)

print("\n--- Duplicate removal results ---")
print("Rows before duplicate removal:", before_dedup)
print("Rows after duplicate removal:", after_dedup)
print("Duplicate rows removed:", before_dedup - after_dedup)

print("\nLabel distribution after duplicate removal:")
print(df_clean["Label"].value_counts())

print("\nRemaining duplicate rows:", df_clean.duplicated().sum())


# Step 8: Save the cleaned dataset
output_path = "data/CIC-IDS2017/Friday-Morning-cleaned.csv"

df_clean.to_csv(output_path, index=False)

print("\nCleaned dataset saved successfully!")
print("File:", output_path)
print("Final dataset shape:", df_clean.shape)
print("Missing values:", df_clean.isnull().sum().sum())
print("Duplicate rows:", df_clean.duplicated().sum())
