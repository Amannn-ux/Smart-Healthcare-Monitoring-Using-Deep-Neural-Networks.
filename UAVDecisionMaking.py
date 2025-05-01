import requests

# ✅ Test different patient conditions
test_cases = [
    {"Temperature_C": 39.5, "Heart_Rate_BPM": 120, "Respiration_Rate": 30, "Blood_Oxygen_Percent": 92},  # Viral Fever
    {"Temperature_C": 36.1, "Heart_Rate_BPM": 80, "Respiration_Rate": 16, "Blood_Oxygen_Percent": 99},  # Healthy
    {"Temperature_C": 38.8, "Heart_Rate_BPM": 110, "Respiration_Rate": 22, "Blood_Oxygen_Percent": 94},  # Jaundice
    {"Temperature_C": 35.0, "Heart_Rate_BPM": 95, "Respiration_Rate": 20, "Blood_Oxygen_Percent": 85}   # Food Poisoning
]

url = "http://127.0.0.1:5000/predict"

for data in test_cases:
    response = requests.post(url, json=data)  # ✅ Ensure JSON format
    print(f"✅ Sent Input: {data}")
    print(f"✅ UAV Response: {response.json()}\n")
