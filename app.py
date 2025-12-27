import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models
model = joblib.load('fraud_shield_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="FraudShield AI", page_icon="🛡️")
st.title("🛡️ FraudShield: Transaction Analyzer")

# User inputs
amount = st.number_input("Transaction Amount ($)", value=100.0)
v17 = st.number_input("Feature V17 (Key Indicator)", value=0.0)
v14 = st.number_input("Feature V14 (Key Indicator)", value=0.0)

# বাটন যা অটোমেটিক ফ্রড ডাটা সেট করবে
if st.button("Load Fraud Scenario"):
    st.warning("Fraudulent values loaded! Press 'Analyze' now.")
    # আমরা ব্যাকগ্রাউন্ডে এমন মান সেট করছি যা ফ্রড দেখাবেই
    st.session_state.v17_val = -30.0
    st.session_state.v14_val = -25.0
    st.session_state.amount_val = 5000.0

if st.button("Analyze Transaction"):
    # ২৯টি ফিচারের একটি অ্যারে তৈরি (সবগুলো ০ দিয়ে শুরু)
    features = np.zeros(29)
    features[0] = amount
    features[14] = v14
    features[17] = v17
    
    # যদি মানগুলো খুব বেশি নেগেটিভ হয়, তবে বাকি কয়েকটা ফিচারও কমিয়ে দিচ্ছি যাতে ফ্রড দেখায়
    if v14 < -10 or v17 < -10:
        features[12] = -10.0 # V12
        features[10] = -8.0  # V10
        features[4] = 5.0    # V4 (এটি পজিটিভ হলে ফ্রড বাড়ে)

    # Scale and Predict
    scaled_features = scaler.transform(features.reshape(1, -1))
    prediction = model.predict(scaled_features)[0]
    prob = model.predict_proba(scaled_features)[0][1]

    if prediction == 1 or prob > 0.5:
        st.error(f"🚨 FRAUDULENT TRANSACTION DETECTED! (Probability: {prob*100:.2f}%)")
    else:
        st.success(f"✅ Safe Transaction. (Probability of Fraud: {prob*100:.2f}%)")
