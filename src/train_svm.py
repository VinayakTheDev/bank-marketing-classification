
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from preprocessing import (
    create_scaled_preprocessor,
    get_features_and_target,
)

df = pd.read_csv("data/raw/bank-full.csv", sep=";")
df = df.drop(columns=["duration"])

X, y = get_features_and_target(df)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Training SVM...")

svm_model = Pipeline([
    ("preprocessor", create_scaled_preprocessor()),
    ("classifier", SVC(
        kernel="rbf",
        class_weight="balanced",
        random_state=42,
    )),
])

svm_model.fit(X_train, y_train)

joblib.dump(svm_model, "models/svm.joblib")

print("SVM saved successfully!")
