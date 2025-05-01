import pandas as pd
import numpy as np

num_patients_per_disease = 1250  # 1250 per disease (Total 5000)

data = []

# ✅ Define controlled feature values per disease
disease_definitions = {
    "Healthy": {"temp_range": (36.0, 37.0), "hr_range": (60, 80), "resp_range": (12, 18), "oxy_range": (97, 100)},
    "Viral Fever": {"temp_range": (38.0, 40.0), "hr_range": (90, 120), "resp_range": (20, 30), "oxy_range": (92, 97)},
    "Jaundice": {"temp_range": (38.5, 40.0), "hr_range": (85, 110), "resp_range": (18, 25), "oxy_range": (95, 99)},
    "Food Poisoning": {"temp_range": (35.5, 37.5), "hr_range": (80, 100), "resp_range": (16, 24), "oxy_range": (85, 95)}
}

for disease, values in disease_definitions.items():
    for _ in range(num_patients_per_disease):
        data.append({
            "Temperature_C": np.random.uniform(*values["temp_range"]),
            "Heart_Rate_BPM": np.random.randint(*values["hr_range"]),
            "Respiration_Rate": np.random.randint(*values["resp_range"]),
            "Blood_Oxygen_Percent": np.random.randint(*values["oxy_range"]),
            "Disease": disease
        })

df = pd.DataFrame(data)
df.to_csv("data/synthetic_patient_data.csv", index=False)
print("✅ Perfectly Balanced Dataset Created: data/synthetic_patient_data.csv")
