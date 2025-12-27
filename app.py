import streamlit as st
import joblib
import numpy as np

# ১. মডেল এবং স্কেলার লোড করা
model = joblib.load('fraud_shield_model.pkl')
scaler = joblib.load('scaler.pkl')

# ২. অ্যাপের ইন্টারফেস ডিজাইন
st.title("🛡️ FraudShield: AI Transaction Security")
st.markdown("Enter transaction details to check for potential fraud.")

# ৩. ইউজার ইনপুট নেওয়া (সহজ করার জন্য আমরা ৩টি গুরুত্বপূর্ণ ইনপুট নিচ্ছি)
amount = st.number_input("Transaction Amount ($)", min_value=0.0)
v17 = st.number_input("Feature V17 (Key Indicator)", value=0.0)
v14 = st.number_input("Feature V14 (Key Indicator)", value=0.0)

# ৪. প্রেডিকশন বাটন
if st.button("Analyze Transaction"):
    # ইনপুট ডাটাকে মডেলের ফরমেটে সাজানো (বাকি ফিচারগুলো ০ ধরে নিচ্ছি উদাহরণের জন্য)
    features = np.zeros(30) 
    features[0] = amount # Scaled amount handling simplifies here
    features[17] = v17
    features[14] = v14
    
    prediction = model.predict([features])
    probability = model.predict_proba([features])[0][1]

    if prediction[0] == 1:
        st.error(f"🚨 ALERT: Potential Fraud Detected! (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Safe Transaction. (Probability of Fraud: {probability:.2%})")
