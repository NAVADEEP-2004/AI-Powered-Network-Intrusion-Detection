
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Project paths
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Page configuration
st.set_page_config(
    page_title="AI-Powered NIDS",
    page_icon="🛡️",
    layout="wide"
)

# Dashboard heading
st.title("🛡️ AI-Powered Network Intrusion Detection System")
st.write(
    "An ML-based prototype for classifying network traffic "
    "as BENIGN or BOT using CIC-IDS2017 data."
)

st.divider()

# Model evaluation results
st.header("Random Forest Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "99.95%")
col2.metric("Balanced Accuracy", "97.69%")
col3.metric("Attack Precision", "99.73%")
col4.metric("Attack Recall", "95.38%")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Attack F1-score")
    st.metric("F1-score", "97.51%")

    st.write(
        "The attack F1-score balances attack precision "
        "and attack recall."
    )

with col2:
    st.subheader("Confusion Matrix")

    matrix_path = ARTIFACTS_DIR / "random_forest_confusion_matrix.png"

    if matrix_path.exists():
        st.image(
            str(matrix_path),
            caption="Random Forest confusion matrix"
        )
    else:
        st.warning(
            "Confusion matrix image not found. "
            "Check the artifacts folder."
        )

st.divider()

# Model comparison
st.header("Model Comparison")

comparison_path = ARTIFACTS_DIR / "model_comparison.csv"
comparison_image = ARTIFACTS_DIR / "model_comparison.png"

if comparison_path.exists():
    comparison_df = pd.read_csv(comparison_path)

    metric_columns = [
        "Accuracy",
        "Balanced Accuracy",
        "Attack Precision",
        "Attack Recall",
        "Attack F1-score",
    ]

    for column in metric_columns:
        if column in comparison_df.columns:
            comparison_df[column] = pd.to_numeric(
                comparison_df[column], errors="coerce"
            ).map(lambda value: f"{value:.2%}" if pd.notna(value) else "N/A")

    st.dataframe(comparison_df, use_container_width=True)
else:
    st.warning("Model comparison CSV was not found.")

if comparison_image.exists():
    st.image(
        str(comparison_image),
        caption="Logistic Regression vs Random Forest"
    )

st.divider()

# Evaluation details
st.header("Evaluation Summary")

evaluation_path = ARTIFACTS_DIR / "final_evaluation.csv"

if evaluation_path.exists():
    evaluation_df = pd.read_csv(evaluation_path)

    for column in metric_columns:
        if column in evaluation_df.columns:
            evaluation_df[column] = pd.to_numeric(
                evaluation_df[column], errors="coerce"
            ).map(lambda value: f"{value:.2%}" if pd.notna(value) else "N/A")

    st.dataframe(evaluation_df, use_container_width=True)
else:
    st.info("Detailed evaluation CSV is not available.")

st.caption(
    "Educational prototype: results reflect the current test split "
    "and do not guarantee real-world intrusion detection performance."
)

import joblib


# Network Traffic Prediction
st.divider()
st.header("🔎 Network Traffic Prediction")

st.write(
    "Choose a sample from the held-out test dataset "
    "and classify its traffic."
)

@st.cache_resource
def load_model():
    return joblib.load(ARTIFACTS_DIR / "random_forest.pkl")


@st.cache_data
def load_test_data():
    sample_df = pd.read_csv(
        ARTIFACTS_DIR / "deployment_sample.csv"
    )
    X_test = sample_df.drop(columns=["label"])
    y_test = sample_df["label"]
    return X_test, y_test


try:
    model = load_model()
    X_test, y_test = load_test_data()

    sample_index = st.number_input(
        "Test sample row",
        min_value=0,
        max_value=len(X_test) - 1,
        value=0,
        step=1
    )

    if st.button("Predict traffic"):
        idx = int(sample_index)

        sample = X_test.iloc[[idx]]
        prediction = model.predict(sample)[0]
        actual_label = y_test.iloc[idx]

        predicted_name = "BOT" if prediction == 1 else "BENIGN"
        actual_name = "BOT" if actual_label == 1 else "BENIGN"

        st.info(f"Model prediction: {predicted_name}")

        st.write(f"**Actual label:** {actual_name}")

        if prediction == actual_label:
            st.success("Correct prediction!")
        else:
            st.warning("Incorrect prediction.")

            if actual_label == 1 and prediction == 0:
                st.error(
                    "False negative: this sample is actually BOT traffic, "
                    "but the model classified it as BENIGN. "
                    "This represents a missed attack."
                )

            elif actual_label == 0 and prediction == 1:
                st.warning(
                    "False positive: this sample is actually BENIGN traffic, "
                    "but the model classified it as BOT."
                )

        st.caption(
            "A BENIGN prediction is not a guarantee that traffic is safe. "
            "This is an educational model evaluated on prepared test samples."
        )

        st.write("Sample features:")
        st.dataframe(
            sample.T.rename(columns={sample.index[0]: "Value"})
        )

except Exception as e:
    st.error(f"Could not load the model or test data: {e}")

# Feature Importance
st.divider()
st.header("📊 Important Features for Detection")

try:
    importance_model = joblib.load(
        ARTIFACTS_DIR / "random_forest.pkl"
    )

    # Handle either a Random Forest model or a Pipeline
    if hasattr(importance_model, "feature_importances_"):
        rf = importance_model
    else:
        rf = importance_model.steps[-1][1]

    importances = rf.feature_importances_

    # Use feature names recorded by the trained model
    
    feature_names = rf.feature_names_in_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    importance_df = importance_df.sort_values(
        "Importance", ascending=False
    ).head(15)

        
    # Keep the original feature names and importance values
    chart_df = importance_df.copy()

    # Shorten labels only for the chart display
    chart_df["Display Feature"] = (
        chart_df["Feature"]
        .str.replace("Init_Win_bytes_backward", "Init Win Bytes Backward", regex=False)
        .str.replace("Init_Win_bytes_forward", "Init Win Bytes Forward", regex=False)
    )

    st.bar_chart(
        chart_df.set_index("Display Feature")["Importance"],
        horizontal=True,
        height=500
    )

    # Show exact names separately for reference
    with st.expander("View complete feature names"):
        st.dataframe(
            importance_df[["Feature", "Importance"]],
            use_container_width=True,
            hide_index=True
        )

    st.caption(
        "These scores indicate the model's relative feature importance; "
        "they do not establish causation."
    )

except Exception as e:
    st.warning(f"Could not display feature importance: {e}")
