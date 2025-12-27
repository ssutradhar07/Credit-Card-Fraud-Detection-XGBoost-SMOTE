import streamlit as st
import pandas as pd
import numpy as np
import joblib

# মডেল এবং স্কেলার লোড করা
try:
    model = joblib.load('fraud_shield_model.pkl')
    scaler = joblib.load('scaler.pkl')
except:
    st.error("Error: Model or Scaler file missing in GitHub!")

st.set_page_config(page_title="FraudShield AI", page_icon="🛡️")

# UI ডিজাইন
st.title("🛡️ FraudShield: AI Transaction Security")
st.markdown("Enter transaction details to check for potential fraud.")

# ইনপুট বক্স
amount = st.number_input("Transaction Amount ($)", value=100.0)
v17 = st.number_input("Feature V17 (Key Indicator)", value=0.0)
v14 = st.number_input("Feature V14 (Key Indicator)", value=0.0)

if st.button("Analyze Transaction"):
    # ২৯টি ফিচারের অ্যারে তৈরি (ডিফল্ট মান ০)
    features = np.zeros(29)
    features[0] = amount
    features[14] = v14
    features[17] = v17
    
    # স্কেলিং এবং প্রেডিকশন
    scaled_features = scaler.transform(features.reshape(1, -1))
    prob = model.predict_proba(scaled_features)[0][1] # ফ্রড হওয়ার সম্ভাবনা
    
    # রেজাল্ট ডিসপ্লে করার লজিক (Custom Threshold)
    # যদি প্রোবাবিলিটি ৫% এর বেশি হয় অথবা কী-ফিচারগুলো খুব নেগেটিভ হয়
    if prob > 0.05 or v14 < -15 or v17 < -15:
        st.error(f"🚨 FRAUDULENT TRANSACTION DETECTED! (Probability: {prob*100:.2f}%)")
        st.info("Technical Note: Extreme negative values in V14/V17 trigger high-risk alerts.")
    else:
        st.success(f"✅ Safe Transaction. (Probability of Fraud: {prob*100:.2f}%)")

st.divider()
st.caption("Disclaimer: This is a simplified demo using XGBoost and SMOTE.")
