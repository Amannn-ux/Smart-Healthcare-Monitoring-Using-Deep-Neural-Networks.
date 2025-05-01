import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# ✅ Load the trained model
model = tf.keras.models.load_model("models/healthcare_model.h5")

# ✅ Load the scaler (Fix "name 'scaler' is not defined" issue)
try:
    scaler = pickle.load(open("models/scaler.pkl", "rb"))
    print("✅ Scaler loaded successfully!")
except FileNotFoundError:
    print("❌ Scaler file not found. Make sure you trained the model correctly!")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print(f"✅ Received Input: {data}")

        sample = pd.DataFrame([data])
        features = ["Temperature_C", "Heart_Rate_BPM", "Respiration_Rate", "Blood_Oxygen_Percent"]
        sample = sample[features]

        # ✅ Apply feature scaling
        sample_scaled = scaler.transform(sample)
        print(f"✅ Scaled Input: {sample_scaled}")

        # ✅ Predict disease using the model
        prediction = model.predict(sample_scaled)
        predicted_disease_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        # ✅ Label Mapping (Ensure Correct Order)
        label_mapping = {0: "Food Poisoning", 1: "Healthy", 2: "Jaundice", 3: "Viral Fever"}
        print(f"✅ Model Prediction: {prediction}")
        print(f"✅ Predicted Disease: {label_mapping[predicted_disease_index]}, Confidence: {confidence}")

        response = {
            "UAV Decision": label_mapping[predicted_disease_index],
            "Confidence": confidence
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
