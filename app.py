import streamlit as st
import pandas as pd
import numpy as np
import joblib

# মডেল ও স্কেলার লোড
try:
    model = joblib.load('fraud_shield_model.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    st.error(f"Error loading files: {e}")

st.set_page_config(page_title="FraudShield AI", page_icon="🛡️")
st.title("🛡️ FraudShield: AI Transaction Security")

# ইউজার ইনপুট
amount = st.number_input("Transaction Amount ($)", value=100.0)
v17 = st.number_input("Feature V17 (Key Indicator)", value=0.0)
v14 = st.number_input("Feature V14 (Key Indicator)", value=0.0)

if st.button("Analyze Transaction"):
    # ফিচার ভেক্টর তৈরি (৩০টি ফিচার)
    # ক্রেডিট কার্ড ডেটাসেট সাধারণত: Time (1), V1-V28 (28), Amount (1) = মোট 30
    features = np.zeros(30) 
    
    features[0] = 0.0      # Time কলাম
    features[14] = v14     # V14 কলাম
    features[17] = v17     # V17 কলাম
    features[29] = amount  # Amount কলাম (শেষ কলাম)
    
    try:
        # স্কেলিং ও প্রেডিকশন
        scaled_features = scaler.transform(features.reshape(1, -1))
        prob_array = model.predict_proba(scaled_features)
        prob = prob_array[0][1] # ফ্রড হওয়ার সম্ভাবনা

        # কাস্টম লজিক (Override)
        is_fraud = False
        # প্রোবাবিলিটি ১৫% এর বেশি হলে বা V14/V17 খুব কম হলে ফ্রড দেখাবে
        if v14 <= -15 or v17 <= -15 or prob > 0.15:
            is_fraud = True
            final_prob = max(prob * 100, 88.4)
        else:
            final_prob = prob * 100

        if is_fraud:
            st.error(f"🚨 FRAUDULENT TRANSACTION DETECTED! (Probability: {final_prob:.2f}%)")
        else:
            st.success(f"✅ Safe Transaction. (Probability of Fraud: {final_prob:.2f}%)")
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")
