import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


filepath = "predictive_maintenance.csv"
df = pd.read_csv(filepath)

df_processed = df.drop(['UDI', 'Product ID'], axis=1)
df_processed = pd.get_dummies(df_processed, columns=['Type'], drop_first=True)

X = df_processed.drop('Target', axis=1)
X = X.drop('Failure Type', axis=1)
y = df_processed['Target']

print("Original shape of features (X):", X.shape)
print("Original distribution of target (y):\n", y.value_counts())
print("-" * 30)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("-" * 30)

numerical_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

scaler = StandardScaler()

X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

print("Data scaled successfully.")
print("Example of scaled data from training set:")
print(X_train.head())
print("-" * 30)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Shape of training data before SMOTE:", X_train.shape)
print("Distribution of target in training data before SMOTE:\n", y_train.value_counts())
print("\nShape of training data after SMOTE:", X_train_resampled.shape)
print("Distribution of target in training data after SMOTE:\n", y_train_resampled.value_counts())
print("-" * 30)
print("Data preparation complete! We are ready to train a model.")