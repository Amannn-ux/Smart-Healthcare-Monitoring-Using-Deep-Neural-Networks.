import tensorflow as tf
import pandas as pd

# ✅ Load the trained model
model = tf.keras.models.load_model("models/healthcare_model.h5")

# ✅ Load test dataset
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()  # Convert DataFrame to Series

# ✅ Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"✅ Model Accuracy on Test Data: {test_accuracy * 100:.2f}%")
