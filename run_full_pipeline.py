import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

filepath = "predictive_maintenance.csv"
df = pd.read_csv(filepath)

headers = ["UDI", "Product ID", "Type",	"Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]",	"Tool wear [min]",	"Target	Failure Type"]

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

num_duplicates = df.duplicated().sum()

print(f"\nFound {num_duplicates} duplicate rows in the dataset.")
print("-" * 50)

df.info()

numerical_summary = df.describe()
print(numerical_summary)

categorical_summary = df.describe(include=['object'])
print(categorical_summary)

print("\n" + "="*80 + "\n")
print("[3] Detailed Breakdown of 'Failure Type':")
failure_type_counts = df['Failure Type'].value_counts()
print(failure_type_counts)

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

print("--- เริ่มกระบวนการสร้างและบันทึกโมเดลที่ดีที่สุด ---")

print("ขั้นตอนที่ 1: กำลังโหลดและเตรียมข้อมูล...")
df_processed = df.drop(['UDI', 'Product ID', 'Failure Type'], axis=1)
df_processed = pd.get_dummies(df_processed, columns=['Type'], drop_first=True)
X = df_processed.drop('Target', axis=1)
y = df_processed['Target']

print("ขั้นตอนที่ 2: กำลังปรับสเกลข้อมูลและสร้างข้อมูลสังเคราะห์...")
numerical_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print(f"ข้อมูลถูกเตรียมพร้อมแล้ว (จำนวน {len(X_resampled)} แถว)")

print("\nขั้นตอนที่ 3: กำลังสอนโมเดล Random Forest ด้วยข้อมูลทั้งหมด...")

final_model = RandomForestClassifier(random_state=42)
final_model.fit(X_resampled, y_resampled)
print("สอนโมเดลสำเร็จ!")

model_filename = "final_random_forest_model.joblib"
scaler_filename = "data_scaler.joblib"

print(f"\nขั้นตอนที่ 4: กำลังบันทึกโมเดลไปที่ไฟล์ '{model_filename}'...")
joblib.dump(final_model, model_filename)
print("บันทึกโมเดลสำเร็จ!")

print(f"กำลังบันทึก Scaler ไปที่ไฟล์ '{scaler_filename}'...")
joblib.dump(scaler, scaler_filename)
print("บันทึก Scaler สำเร็จ!")

print("\n--- กระบวนการเสร็จสมบูรณ์ ---")
print("ตอนนี้คุณมีไฟล์โมเดลและไฟล์ Scaler พร้อมสำหรับนำไปใช้งานจริงแล้ว")

print("ขั้นตอนที่ 1: กำลังโหลดโมเดลและ Scaler...")
model = joblib.load("final_random_forest_model.joblib")
scaler = joblib.load("data_scaler.joblib")
print("โหลดสำเร็จ!")

new_data_normal = {
    'Type': 'L',
    'Air temperature [K]': 300.1,
    'Process temperature [K]': 309.8,
    'Rotational speed [rpm]': 1500,
    'Torque [Nm]': 45.3,
    'Tool wear [min]': 108
}

new_data_suspicious = {
    'Type': 'L',
    'Air temperature [K]': 303.5,
    'Process temperature [K]': 311.8,
    'Rotational speed [rpm]': 1380, 
    'Torque [Nm]': 65.7,           
    'Tool wear [min]': 215          
}

data_to_predict = new_data_suspicious
print(f"\nขั้นตอนที่ 2: ข้อมูลใหม่ที่ต้องการทำนาย:\n{data_to_predict}")

print("\nขั้นตอนที่ 3: กำลังเตรียมข้อมูลใหม่...")

df_new = pd.DataFrame([data_to_predict])

df_new['Type_L'] = df_new['Type'] == 'L'
df_new['Type_M'] = df_new['Type'] == 'M'
df_new = df_new.drop('Type', axis=1)

numerical_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
df_new[numerical_features] = scaler.transform(df_new[numerical_features])

model_feature_order = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Type_L', 'Type_M']
df_new = df_new[model_feature_order]

print("เตรียมข้อมูลใหม่สำเร็จ!")
print("ข้อมูลหลังการแปลง:\n", df_new.head())

print("\nขั้นตอนที่ 4: กำลังทำนายผล...")
prediction = model.predict(df_new)
prediction_proba = model.predict_proba(df_new)

print("\n--- ผลการทำนาย ---")
if prediction[0] == 1:
    print("ผลลัพธ์: [1] - มีความเสี่ยงที่จะเกิดความเสียหาย (Failure is likely)")
else:
    print("ผลลัพธ์: [0] - เครื่องจักรทำงานปกติ (Machine is stable)")

print(f"\nความน่าจะเป็นในการทำนาย:")
print(f" - โอกาสที่จะไม่พัง (Class 0): {prediction_proba[0][0]:.2%}")
print(f" - โอกาสที่จะพัง (Class 1): {prediction_proba[0][1]:.2%}")

print("--- Preparing Data for Power BI ---")

print("Step 1: Loading model and scaler...")
model = joblib.load("final_random_forest_model.joblib")
scaler = joblib.load("data_scaler.joblib")
print("Load successful!")

print("Step 2: Loading original dataset...")
df_original = pd.read_csv("predictive_maintenance.csv")

print("Step 3: Preparing data for prediction...")
df_to_predict = df_original.copy()

df_for_model = df_to_predict.drop(['UDI', 'Product ID', 'Failure Type', 'Target'], axis=1) 
df_for_model = pd.get_dummies(df_for_model, columns=['Type'], drop_first=True)

expected_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Type_L', 'Type_M']
for col in expected_cols:
    if col not in df_for_model.columns:
        df_for_model[col] = False

numerical_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
df_for_model[numerical_features] = scaler.transform(df_for_model[numerical_features])

df_for_model = df_for_model[expected_cols]

print("Step 4: Making predictions on the entire dataset...")
predictions = model.predict(df_for_model)
probabilities = model.predict_proba(df_for_model)[:, 1] 

df_original['Predicted Failure'] = predictions
df_original['Prediction Confidence (%)'] = probabilities * 100
df_original['Prediction Confidence (%)'] = df_original['Prediction Confidence (%)'].round(2)

df_original['Prediction Result'] = df_original['Predicted Failure'].apply(lambda x: 'Failure Likely' if x == 1 else 'Stable')

output_filename = 'powerbi_dashboard_data.csv'
print(f"Step 5: Saving data to '{output_filename}'...")
df_original.to_csv(output_filename, index=False, encoding='utf-8')
print(f"Successfully created '{output_filename}' for Power BI.")

print("\nStep 6: Creating a simulated prediction log...")
log_data = {
    'Timestamp': pd.to_datetime(['2025-06-19 08:50:01', '2025-06-19 08:51:15', '2025-06-19 08:52:30']),
    'Machine_ID': ['M-001', 'M-002', 'M-001'],
    'Torque [Nm]': [45.1, 65.7, 70.2],
    'Tool_Wear [min]': [10, 215, 225],
    'Prediction': ['Stable', 'Failure Likely', 'Failure Likely'],
    'Confidence': [5.2, 96.0, 98.5]
}
log_df = pd.DataFrame(log_data)
log_filename = 'powerbi_prediction_log.csv'
log_df.to_csv(log_filename, index=False, encoding='utf-8')
print(f"Successfully created '{log_filename}' for simulating live data.")

print("\n--- All files are ready for Power BI! ---")