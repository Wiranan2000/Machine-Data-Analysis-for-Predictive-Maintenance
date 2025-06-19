import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

filepath = "predictive_maintenance.csv"
df = pd.read_csv(filepath)

sns.set_style("whitegrid") 
plt.rcParams['figure.figsize'] = (16, 10) 

numerical_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

print("Generating histograms for numerical features...")
df[numerical_features].hist(bins=30, edgecolor='black')
plt.suptitle("Distribution of Numerical Features", size=20)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

categorical_features = ['Type', 'Failure Type']
fig, axes = plt.subplots(1, 2, figsize=(16, 6)) 

sns.countplot(x='Type', data=df, ax=axes[0], order=df['Type'].value_counts().index)
axes[0].set_title('Machine Type Distribution')

sns.countplot(y='Failure Type', data=df, ax=axes[1], order=df['Failure Type'].value_counts().index)
axes[1].set_title('Failure Type Distribution')

plt.tight_layout()
plt.show() 