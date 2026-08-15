"""
app.py -- Streamlit demo app for ML Assignment 2
Breast Cancer Wisconsin (Diagnostic) classification models

Features required by the assignment:
  a. Dataset upload option (CSV)               -> st.file_uploader
  b. Model selection dropdown                  -> st.selectbox
  c. Display of evaluation metrics              -> metrics table
  d. Confusion matrix / classification report   -> heatmap + text report
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

st.set_page_config(page_title="ML Assignment 2 - Classifier Demo", layout="wide")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest (Ensemble)": "model/random_forest_ensemble.pkl",
}


@st.cache_resource
def load_scaler():
    return joblib.load("model/scaler.pkl")


@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_feature_names():
    with open("model/feature_names.json") as f:
        return json.load(f)


st.title("Breast Cancer Classification - Model Comparison App")
st.caption(
    "M.Tech (AIML/DSE) - Machine Learning - Assignment 2 | "
    "Dataset: Breast Cancer Wisconsin (Diagnostic)"
)

# ---------------------------------------------------------------------------
# Sidebar: model selection
# ---------------------------------------------------------------------------
st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox("Select a model", list(MODEL_FILES.keys()))

# ---------------------------------------------------------------------------
# Main: dataset upload
# ---------------------------------------------------------------------------
st.subheader("1. Upload Test Data (CSV)")
st.write(
    "Upload `test_data.csv` (or any CSV with the same 30 feature columns "
    "and a `target` column: 0 = malignant, 1 = benign)."
)
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    feature_names = load_feature_names()

    if "target" not in df.columns:
        st.error("Uploaded CSV must include a 'target' column with true labels.")
    elif not set(feature_names).issubset(df.columns):
        missing = set(feature_names) - set(df.columns)
        st.error(f"Uploaded CSV is missing required feature columns: {missing}")
    else:
        X = df[feature_names]
        y_true = df["target"]

        scaler = load_scaler()
        X_scaled = scaler.transform(X)

        model = load_model(MODEL_FILES[model_choice])
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)[:, 1]

        # -------------------------------------------------------------
        # 2. Evaluation metrics
        # -------------------------------------------------------------
        st.subheader(f"2. Evaluation Metrics - {model_choice}")

        try:
            auc = roc_auc_score(y_true, y_proba)
        except ValueError:
            auc = float("nan")  # happens if uploaded data has only one class

        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "AUC": auc,
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1 Score": f1_score(y_true, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y_true, y_pred),
        }
        metrics_df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
        metrics_df["Value"] = metrics_df["Value"].round(4)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.table(metrics_df.set_index("Metric"))
        with col2:
            st.bar_chart(metrics_df.set_index("Metric"))

        # -------------------------------------------------------------
        # 3. Confusion matrix + classification report
        # -------------------------------------------------------------
        st.subheader("3. Confusion Matrix & Classification Report")
        c1, c2 = st.columns(2)

        with c1:
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Malignant", "Benign"],
                yticklabels=["Malignant", "Benign"], ax=ax
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)

        with c2:
            report = classification_report(y_true, y_pred, target_names=["Malignant", "Benign"])
            st.text("Classification Report")
            st.code(report)

        # -------------------------------------------------------------
        # 4. All-model comparison (pre-computed on the original test split)
        # -------------------------------------------------------------
        st.subheader("4. All-Model Comparison (reference, from training run)")
        try:
            comparison_df = pd.read_csv("model/metrics_comparison.csv")
            st.dataframe(comparison_df, use_container_width=True)
        except FileNotFoundError:
            st.info("Run train_models.py to generate model/metrics_comparison.csv")

else:
    st.info("Upload a CSV file above to see predictions and metrics.")

st.divider()
st.caption("Built with Streamlit • scikit-learn models trained in train_models.py")
