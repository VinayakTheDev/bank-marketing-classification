import pandas as pd

DATA_PATH = "data/raw/bank-full.csv"

# Bank Marketing CSV uses semicolon delimiters
df = pd.read_csv(DATA_PATH, sep=";")

print("\n========== DATASET OVERVIEW ==========")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== COLUMN INFORMATION ==========")
print(df.info())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== TARGET DISTRIBUTION ==========")
print(df["y"].value_counts())
print("\nTarget percentage:")
print(df["y"].value_counts(normalize=True) * 100)

print("\n========== NUMERICAL FEATURES ==========")
print(df.describe())

print("\n========== CATEGORICAL FEATURES ==========")
print(df.describe(include="object"))

print("\n========== FEATURE COUNT ==========")
print(f"Total columns: {df.shape[1]}")
print(f"Target column: y")
print(f"Predictor columns including duration: {df.shape[1] - 1}")
print(f"Predictor columns excluding duration: {df.shape[1] - 2}")
