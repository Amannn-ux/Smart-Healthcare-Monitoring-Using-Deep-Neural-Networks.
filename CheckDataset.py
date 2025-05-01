import pandas as pd

df = pd.read_csv("data/synthetic_patient_data.csv")

# ✅ Print disease counts
print("\n✅ Disease Distribution in Dataset:")
print(df["Disease"].value_counts())

# ✅ Check feature averages per disease
print("\n✅ Average Feature Values per Disease:")
print(df.groupby("Disease").mean())
