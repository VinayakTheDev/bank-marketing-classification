import pandas as pd

from sklearn.model_selection import train_test_split
from preprocessing import get_features_and_target

# Load original dataset
df = pd.read_csv("data/raw/bank-full.csv", sep=";")

# Exclude duration to prevent leakage
df = df.drop(columns=["duration"])

# Prepare features and target
X, y = get_features_and_target(df)

# Recreate the exact train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Save test data
test_data = X_test.copy()
test_data["y"] = y_test.map({0: "no", 1: "yes"})

test_data.to_csv("test_data.csv", index=False)

print("Test data created successfully!")
print("Rows:", len(test_data))
print("Columns:", len(test_data.columns))
print("Saved to test_data.csv")
