import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

filepath = "predictive_maintenance.csv"
df = pd.read_csv(filepath)

print("Step 1: Loading and Preparing Data...")
filepath = "predictive_maintenance.csv"
df = pd.read_csv(filepath)
df_processed = df.drop(['UDI', 'Product ID', 'Failure Type'], axis=1)
df_processed = pd.get_dummies(df_processed, columns=['Type'], drop_first=True)
X = df_processed.drop('Target', axis=1)
y = df_processed['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
numerical_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("Data Preparation Complete.\n" + "="*50 + "\n")

models = {"Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)}
for name, model in models.items():
    print(f"Step 2: Training {name}...")
    
    model.fit(X_train_resampled, y_train_resampled)
    print(f"{name} trained successfully.")

    print(f"\nStep 3: Evaluating {name} on the Test Set...")
   
    y_pred = model.predict(X_test)
    print(f"\n--- Results for {name} ---")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Failure (0)', 'Failure (1)'], zero_division=0))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted No Failure', 'Predicted Failure'], yticklabels=['Actual No Failure', 'Actual Failure'])
    plt.title(f'Confusion Matrix for {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    print("\n" + "="*50 + "\n")