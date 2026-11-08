
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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

FEATURES = [
    "age", "balance", "day", "campaign", "pdays", "previous",
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "poutcome"
]

st.set_page_config(
    page_title="Bank Marketing ML Evaluation",
    page_icon="📊",
    layout="wide"
)

st.title("Bank Marketing Classification")
st.write("Compare five machine learning classifiers using a test dataset.")

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib",
}

selected_model = st.selectbox(
    "Select Model",
    list(model_files.keys())
)

uploaded_file = st.file_uploader(
    "Upload test CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    required_columns = FEATURES + ["y"]
    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        st.error(f"Missing columns: {missing_columns}")
        st.stop()

    X = df[FEATURES].copy()

    y = df["y"].map({"no": 0, "yes": 1})

    if y.isna().any():
        st.error("Target column must contain only 'yes' or 'no'.")
        st.stop()

    model_path = MODEL_DIR / model_files[selected_model]

    if not model_path.exists():
        st.error(f"Model not found: {model_path}")
        st.stop()

    model = joblib.load(model_path)

    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    else:
        y_score = model.decision_function(X)

    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1 Score": f1_score(y, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y, y_score),
        "MCC": matthews_corrcoef(y, y_pred),
    }

    st.subheader(f"Evaluation Results: {selected_model}")

    cols = st.columns(3)

    for i, (name, value) in enumerate(metrics.items()):
        cols[i % 3].metric(name, f"{value:.4f}")

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    st.subheader("Classification Report")

    report = classification_report(
        y,
        y_pred,
        target_names=["No", "Yes"],
        output_dict=True,
        zero_division=0
    )

    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Predictions")

    predictions = df.copy()
    predictions["actual"] = y.map({0: "no", 1: "yes"})
    predictions["predicted"] = pd.Series(y_pred).map(
        {0: "no", 1: "yes"}
    ).values

    st.dataframe(predictions.head(100))

    st.download_button(
        "Download Predictions",
        predictions.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv"
    )

else:
    st.info("Upload test_data.csv to evaluate the selected model.")
