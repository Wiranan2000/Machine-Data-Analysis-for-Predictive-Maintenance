import pandas as pd

filepath = "predictive_maintenance.csv"
df = pd.read_csv(filepath)

numerical_summary = df.describe()
print(numerical_summary)

categorical_summary = df.describe(include=['object'])
print(categorical_summary)

print("\n" + "="*80 + "\n")
print("[3] Detailed Breakdown of 'Failure Type':")
failure_type_counts = df['Failure Type'].value_counts()
print(failure_type_counts)