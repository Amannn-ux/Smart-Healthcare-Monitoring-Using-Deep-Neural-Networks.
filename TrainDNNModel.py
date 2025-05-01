import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

# ✅ Load dataset
df = pd.read_csv("data/synthetic_patient_data.csv")

# ✅ Encode disease labels
label_encoder = LabelEncoder()
df["Disease"] = label_encoder.fit_transform(df["Disease"])
print("✅ Label Mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# ✅ Define feature columns
features = ["Temperature_C", "Heart_Rate_BPM", "Respiration_Rate", "Blood_Oxygen_Percent"]

# ✅ Apply feature scaling
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])  

# ✅ Save scaler for FlaskServer.py
pickle.dump(scaler, open("models/scaler.pkl", "wb"))
print("✅ Scaler saved as 'models/scaler.pkl'")

# ✅ Split data into training & testing sets
X = df[features]
y = df["Disease"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ✅ Define the neural network model (Reduced Overfitting)
model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.5),  # ✅ Increased dropout to prevent memorization
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(len(y_train.unique()), activation='softmax')  # ✅ Predicts multiple classes
])

# ✅ Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ✅ Train the model (Reduced epochs to prevent overfitting)
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2)

# ✅ Save trained model
model.save("models/healthcare_model.h5")
print("✅ Model saved as 'models/healthcare_model.h5'")

# ✅ Evaluate model on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"✅ Model Accuracy on Test Data: {test_accuracy * 100:.2f}%")
