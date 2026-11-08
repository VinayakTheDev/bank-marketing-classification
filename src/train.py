
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from preprocessing import (
    create_scaled_preprocessor,
    create_unscaled_preprocessor,
    get_features_and_target,
)

DATA_PATH = "data/raw/bank-full.csv"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# 1. Load dataset
df = pd.read_csv(DATA_PATH, sep=";")

# 2. Remove duration to prevent target leakage
df = df.drop(columns=["duration"])

# 3. Separate features and target
X, y = get_features_and_target(df)

# 4. Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Training target distribution:")
print(y_train.value_counts(normalize=True))

# 5. Define models
models = {
    "logistic_regression": Pipeline([
        ("preprocessor", create_scaled_preprocessor()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ]),

    "decision_tree": Pipeline([
        ("preprocessor", create_unscaled_preprocessor()),
        ("classifier", DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42,
        )),
    ]),

    "knn": Pipeline([
        ("preprocessor", create_scaled_preprocessor()),
        ("classifier", KNeighborsClassifier(n_neighbors=5)),
    ]),

    "naive_bayes": Pipeline([
        ("preprocessor", create_unscaled_preprocessor()),
        ("classifier", GaussianNB()),
    ]),

    "svm": Pipeline([
        ("preprocessor", create_scaled_preprocessor()),
        ("classifier", SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=42,
        )),
    ]),
}

# 6. Train and save
for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    output_path = MODEL_DIR / f"{name}.joblib"
    joblib.dump(model, output_path)

    print(f"Saved: {output_path}")

# 7. Save test data for evaluation and Streamlit
test_data = X_test.copy()
test_data["y"] = y_test.map({0: "no", 1: "yes"})
test_data.to_csv("test_data.csv", index=False)

print("\nTraining complete.")
print("Saved models:", list(MODEL_DIR.glob("*.joblib")))
print("Saved test_data.csv")
