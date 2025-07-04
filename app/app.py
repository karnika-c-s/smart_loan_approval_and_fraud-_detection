import streamlit as st
import joblib
import pandas as pd

# Load the model pipeline
model = joblib.load('../models/loan_classifier_binary.pkl')

st.set_page_config(page_title="Smart Loan Predictor", layout="centered")
st.title("🏦 Smart Loan Approval Predictor (Minimal Inputs)")

# 🔘 Minimal user inputs
monthly_income = st.number_input("Monthly Income (₹)", min_value=0, value=30000)
loan_amount = st.number_input("Loan Amount Requested (₹)", min_value=0, value=50000)
cibil_score = st.slider("CIBIL Score", 300, 900, 750)
employment_status = st.selectbox("Employment Status", ['Employed', 'Unemployed', 'Self-employed', 'Retired'])

# 🎯 Remaining inputs filled with default values (hidden from UI)
defaults = {
    'applicant_age': 35,
    'loan_tenure_months': 60,
    'interest_rate_offered': 11.5,
    'existing_emis_monthly': 5000,
    'debt_to_income_ratio': 0.3,
    'number_of_dependents': 2,
    'avg_txn_amount': 15000,
    'total_txn_amount': 250000,
    'txn_count': 12,
    'failed_txn_ratio': 0.05,
    'intl_txn_count': 1,
    'gender': 'Male',
    'loan_type': 'Personal',
    'property_ownership_status': 'Owned',
    'purpose_of_loan': 'Medical',
    'most_common_txn_type': 'UPI',
    'most_used_device': 'Mobile'
}

# Combine into full input row
input_data = pd.DataFrame([{
    'monthly_income': monthly_income,
    'loan_amount_requested': loan_amount,
    'cibil_score': cibil_score,
    'employment_status': employment_status,
    **defaults  # Inject default values for all other expected features
}])

# Predict on button click
if st.button("📊 Predict Loan Status"):
    try:
        prediction = model.predict(input_data)[0]
        result = "✅ Loan Approved" if prediction == 0 else "❌ Loan Declined or Fraud Risk"
        st.success(result)
    except Exception as e:
        st.error("Prediction failed. Check input structure.")
        st.exception(e)
