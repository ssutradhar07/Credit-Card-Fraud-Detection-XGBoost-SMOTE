# 🛡️ FraudShield: AI-Powered Transaction Security System

FraudShield is a high-performance machine learning application designed to detect fraudulent credit card transactions in real-time. This project handles extreme class imbalance and provides explainable AI insights for every prediction.

## 🚀 Live Demo
**Check out the live application here:** [FraudShield Live App](https://credit-card-fraud-detection-xgboost-smote-gubtqkspscnmybjxltrb.streamlit.app/)

---

## 📊 Key Features
- **Real-time Prediction:** Instant classification of transactions as 'Legit' or 'Fraud'.
- **Class Imbalance Handling:** Utilized **SMOTE** (Synthetic Minority Over-sampling Technique) to ensure the model learns from rare fraud cases.
- **High Performance:** Powered by **XGBoost**, achieving high Recall to minimize financial risks.
- **Explainable AI:** Integrated **SHAP** values to help users understand why a transaction was flagged as fraudulent.

## 🛠️ Tech Stack
- **Language:** Python 3.11
- **Modeling:** XGBoost, Scikit-learn, Imbalanced-learn
- **Dashboard:** Streamlit
- **Explainability:** SHAP
- **Deployment:** Streamlit Cloud

## 📂 Repository Content
- `app.py`: The core Streamlit application.
- `fraud_shield_model.pkl`: The trained XGBoost model.
- `scaler.pkl`: RobustScaler object for data normalization.
- `requirements.txt`: List of dependencies for environment setup.
- `FraudShield_MLOps_Project.ipynb`: Full data analysis and training notebook.

## ⚙️ Local Installation
1. Clone this repository:
   ```bash
   git clone [https://github.com/ssutradhar07/Credit-Card-Fraud-Detection-XGBoost-SMOTE.git](https://github.com/ssutradhar07/Credit-Card-Fraud-Detection-XGBoost-SMOTE.git)



   Install dependencies:
   pip install -r requirements.txt

   Run the app:
   streamlit run app.py

   Developed by: [Shovona Sutradhar]
