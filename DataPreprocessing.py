import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("data/synthetic_patient_data.csv")

# Encode categorical labels
label_encoder = LabelEncoder()
df["Disease"] = label_encoder.fit_transform(df["Disease"])

# Scale features correctly
features = ["Temperature_C", "Heart_Rate_BPM", "Respiration_Rate", "Blood_Oxygen_Percent"]
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])  

# Save preprocessed data
df.to_csv("data/processed_data.csv", index=False)
print("✅ Data preprocessing complete: data/processed_data.csv")
