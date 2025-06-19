import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

filepath = "predictive_maintenance.csv"
df = pd.read_csv(filepath)

sns.set_style("whitegrid")
numerical_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

print("Generating box plots to compare features between Failure and No Failure cases...")

for feature in numerical_features:
    plt.figure(figsize=(8, 6)) 
    sns.boxplot(
        x='Target',  
        y=feature,   
        data=df
    )
    plt.title(f'{feature} vs. Failure Status', fontsize=14)
    plt.xticks([0, 1], ['No Failure', 'Failure']) 
    plt.show()