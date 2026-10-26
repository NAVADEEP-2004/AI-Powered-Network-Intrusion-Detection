
import pandas as pd

file_path = "data/CIC-IDS2017/Friday-WorkingHours-Morning.pcap_ISCX.csv"

df = pd.read_csv(file_path)

# Remove accidental spaces around column names
df.columns = df.columns.str.strip()

print("1. Dataset shape:")
print(df.shape)

print("\n2. Label distribution:")
print(df["Label"].value_counts())

print("\n3. Missing values:")
print(df.isnull().sum().sum())

print("\n4. Duplicate rows:")
print(df.duplicated().sum())

print("\n5. Data types:")
print(df.dtypes.value_counts())

print("\n6. Infinite values in numeric columns:")
numeric_df = df.select_dtypes(include="number")
print(numeric_df.isin([float("inf"), float("-inf")]).sum().sum())


# 7. Missing values by column
print("\n7. Columns containing missing values:")
missing = df.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False))

# 8. Infinite values by column
print("\n8. Columns containing infinite values:")
numeric_df = df.select_dtypes(include="number")
infinite = numeric_df.isin([float("inf"), float("-inf")]).sum()
print(infinite[infinite > 0].sort_values(ascending=False))

# 9. Duplicate records by label
print("\n9. Labels among duplicate rows:")
print(df[df.duplicated(keep=False)]["Label"].value_counts())
