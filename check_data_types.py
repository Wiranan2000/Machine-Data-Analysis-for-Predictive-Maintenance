import pandas as pd

filepath = "predictive_maintenance.csv"
df = pd.read_csv(filepath)

df.info()