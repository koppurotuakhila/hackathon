import pandas as pd

# Load dataset
df = pd.read_csv("dataset/cicids2017_cleaned.csv")

print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nUnique Labels:")
print(df["Attack Type"].value_counts())