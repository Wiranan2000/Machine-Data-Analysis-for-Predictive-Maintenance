import pandas as pd

filepath = "predictive_maintenance.csv"
df = pd.read_csv(filepath)

missing_per_column = df.isnull().sum()
print(missing_per_column)
print("-" * 50)

total_missing = df.isnull().sum().sum()
print(f"Total missing values in the entire dataset: {total_missing}")
print("-" * 50)

rows_with_missing_data = df[df.isnull().any(axis=1)]
if rows_with_missing_data.empty:
    print("No rows with missing values found. The dataset is complete!")
else:
    print(rows_with_missing_data)
print("-" * 50)