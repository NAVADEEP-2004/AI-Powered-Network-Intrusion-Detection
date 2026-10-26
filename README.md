# AI-Powered Network Intrusion Detection System

## Overview
An ML-based Network Intrusion Detection System (NIDS) that classifies network traffic as BENIGN or BOT using features from the CIC-IDS2017 dataset.

The project compares Logistic Regression and Random Forest models and provides an interactive Streamlit dashboard for model evaluation and traffic prediction.

## Features
- Network traffic data inspection and preprocessing
- Feature preparation and train-test splitting
- Logistic Regression and Random Forest model training
- Model comparison using classification metrics
- Confusion matrix and feature importance visualization
- Interactive Streamlit dashboard for test-sample predictions
- Basic data leakage checks

## Technology Stack
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Streamlit
- Joblib

## Evaluation Metrics
The current Random Forest test results are:

| Metric | Score |
|---|---:|
| Accuracy | 99.95% |
| Balanced Accuracy | 97.69% |
| Attack Precision | 99.73% |
| Attack Recall | 95.38% |
| Attack F1-score | 97.51% |

These are results from the current test split, not a guarantee of real-world performance. Further validation is required before practical deployment.

## Run Locally

1. Clone or download this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   `python -m pip install -r requirements.txt`

4. Ensure the required dataset and generated model artifacts are available in their expected folders.
5. Start the dashboard:

   `python -m streamlit run app.py`

## Project Status
Educational prototype in development.

## Disclaimer
This project is intended for educational and research purposes. Predictions should not be treated as definitive evidence of malicious activity or used as the sole basis for security decisions.