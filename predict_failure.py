import pandas as pd
import joblib

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