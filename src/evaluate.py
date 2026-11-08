
import pandas as pd
import joblib

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

MODEL_DIR = Path("models")

# Load test dataset
test_df = pd.read_csv("test_data.csv")

X_test = test_df.drop(columns=["y"])
y_test = test_df["y"].map({"no": 0, "yes": 1})

results = []

for model_path in sorted(MODEL_DIR.glob("*.joblib")):
    model_name = model_path.stem
    model = joblib.load(model_path)

    print(f"\n{'=' * 60}")
    print(f"MODEL: {model_name.upper()}")
    print("=" * 60)

    y_pred = model.predict(X_test)
   # Obtain prediction scores for ROC-AUC
    if hasattr(model, "predict_proba"):
     y_score = model.predict_proba(X_test)[:, 1]
    else:
     y_score = model.decision_function(X_test)

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_score),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }

    results.append(metrics)

    print("\nMetrics:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["no", "yes"],
        zero_division=0,
    ))

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("F1", ascending=False)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

results_df.to_csv("model_comparison.csv", index=False)

print("\nSaved model_comparison.csv")
