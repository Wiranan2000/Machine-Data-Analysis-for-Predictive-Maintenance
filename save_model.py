import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import joblib 

filepath = "predictive_maintenance.csv"
df = pd.read_csv(filepath)

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