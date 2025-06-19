import pandas as pd
import joblib
import os

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