
import pandas as pd

# Location of our dataset
file_path = "data/CIC-IDS2017/Friday-WorkingHours-Morning.pcap_ISCX.csv"

# Load the CSV file
df = pd.read_csv(file_path)

# Display basic information
print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()
