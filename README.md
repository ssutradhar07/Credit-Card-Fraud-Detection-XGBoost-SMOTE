# Credit-Card-Fraud-Detection-XGBoost-SMOTE
End-to-end MLOps system to detect credit card fraud using XGBoost. Solves extreme class imbalance with SMOTE and provides model transparency through SHAP (Explainable AI). 🚀 Deployed on Streamlit.

# 🛡️ FraudShield: AI-Powered Transaction Security

FraudShield is a machine learning-based system designed to detect fraudulent credit card transactions in real-time. This project addresses the challenge of highly imbalanced data to provide accurate and explainable security insights.

## 🚀 Project Overview
- **Objective:** To identify fraudulent transactions from credit card data.
- **Model:** XGBoost Classifier.
- **Handling Imbalance:** Used **SMOTE** (Synthetic Minority Over-sampling Technique).
- **Explainability:** Integrated **SHAP** to understand which features trigger fraud alerts.

## 🛠️ Tech Stack
- **Language:** Python
- **ML Libraries:** Scikit-learn, XGBoost, Imbalanced-learn
- **Dashboard:** Streamlit
- **Deployment:** Streamlit Cloud

## 📂 Repository Structure
- `app.py`: The web application code.
- `fraud_shield_model.pkl`: The trained machine learning model.
- `scaler.pkl`: Pre-processing scaler for data normalization.
- `requirements.txt`: List of necessary Python libraries.

## ⚙️ How to Run
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Launch the app: `streamlit run app.py`

---
*Developed as an End-to-End MLOps Project.*
