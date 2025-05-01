import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
import pandas as pd
from tensorflow.keras.utils import to_categorical

# Load your trained model
model = load_model(r"C:\Users\KIIT\Desktop\smart healthcare project\models\healthcare_model.h5")

# Load testing data (X_test and y_test CSV files)
X_test = pd.read_csv(r"C:\Users\KIIT\Desktop\smart healthcare project\data\X_test.csv").values  # Load and convert to NumPy array
y_test = pd.read_csv(r"C:\Users\KIIT\Desktop\smart healthcare project\data\y_test.csv").values  # Load and convert to NumPy array

# If y_test is categorical, one-hot encode it
y_test = to_categorical(y_test)

# Remove "Healthy Patient" (class 0)
X_test = X_test[y_test[:, 0] != 1]  # Remove rows where "Healthy Patient" is present
y_test = y_test[y_test[:, 0] != 1]  # Remove corresponding labels

# Get predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Compute confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)

# Define the remaining class names (after removing "Healthy Patient")
class_names = ["Food Poisoning", "Jaundice", "Viral Fever", "Heart Disease"]

# Reverse the order of the class names (optional)
reversed_class_names = class_names[::-1]

# Plot confusion matrix with reversed order and color enhancements
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',  # Changed color palette to 'YlGnBu' (Yellow-Green-Blue)
            xticklabels=reversed_class_names, yticklabels=reversed_class_names, 
            linewidths=0.5, linecolor='black',  # Add grid lines for clarity
            cbar_kws={'label': 'Count'},  # Add a colorbar with label
            square=True)  # Make the plot square to look more balanced
plt.xlabel("Predicted Label", fontsize=12, weight='bold')
plt.ylabel("True Label", fontsize=12, weight='bold')
plt.title("Confusion Matrix for DL Model", fontsize=14, weight='bold')
plt.show()

# Print classification report
print("Classification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=reversed_class_names))
