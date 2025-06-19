import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

filepath = "predictive_maintenance.csv"
df = pd.read_csv(filepath)

correlation_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
df_corr = df[correlation_features]
corr_matrix = df_corr.corr()

print("Generating correlation heatmap...")
plt.figure(figsize=(10, 8)) 
sns.heatmap(
    corr_matrix,    
    annot=True,      
    cmap='coolwarm',
    fmt=".2f")
plt.title('Correlation Matrix of Numerical Features', fontsize=16)
plt.show()