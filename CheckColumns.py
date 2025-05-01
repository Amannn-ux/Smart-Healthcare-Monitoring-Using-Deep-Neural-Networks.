import pandas as pd

# Load dataset
df = pd.read_csv("data/synthetic_patient_data.csv")

# Print column names
print("Column Names in CSV:", df.columns.tolist())

# Show first few rows
print(df.head())
