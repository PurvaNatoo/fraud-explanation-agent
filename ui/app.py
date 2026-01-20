import streamlit as st
import pandas as pd
from agent.fraud_agent import agent_decision

st.title("Fraud Detection Agent")

st.header("1. Enter Application Details")

with st.form("transaction_form"):
    customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=30)
    date_of_birth_distinct_emails_4w = st.number_input("Distinct Emails 4W", value=0)
    income = st.slider("Income (0-1)", 0.0, 1.0, 0.5)
    credit_risk_score = st.number_input("Credit Risk Score", value=600)
    proposed_credit_limit = st.number_input("Proposed Credit Limit", value=1000)
    has_other_cards = st.checkbox("Has Other Cards")
    
    prev_address_months_count = st.number_input("Previous Address Months", value=12)
    current_address_months_count = st.number_input("Current Address Months", value=24)
    days_since_request = st.number_input("Days Since Request", value=0)
    intended_balcon_amount = st.number_input("Intended Balance Amount", value=500)
    zip_count_4w = st.number_input("Applications in ZIP (4W)", value=1)
    bank_branch_count_8w = st.number_input("Bank Branch Count (8W)", value=0)
    foreign_request = st.checkbox("Foreign Request")
    session_length_in_minutes = st.number_input("Session Length (min)", value=10)
    keep_alive_session = st.checkbox("Keep Alive Session")
    device_distinct_emails_8w = st.number_input("Device Distinct Emails (8W)", value=0)
    device_fraud_count = st.number_input("Device Fraud Count", value=0)
    month = st.number_input("Month", min_value=1, max_value=12, value=1)
    bank_months_count = st.number_input("Bank Months Count", value=12)

    st.header("Validation / Flags")
    email_is_free = st.checkbox("Email is free")
    phone_home_valid = st.checkbox("Phone Home Valid")
    phone_mobile_valid = st.checkbox("Phone Mobile Valid")

    st.header("Velocity / Behavior")
    velocity_6h = st.number_input("Velocity 6h", value=0)
    velocity_24h = st.number_input("Velocity 24h", value=0)
    velocity_4w = st.number_input("Velocity 4w", value=0)

    submitted = st.form_submit_button("Compute Decision")

if submitted:
    user_dict = {
        "income": income,
        "name_email_similarity": 0.5,  # or another default if not collected from UI
        "prev_address_months_count": prev_address_months_count,
        "current_address_months_count": current_address_months_count,
        "customer_age": customer_age,
        "days_since_request": days_since_request,
        "intended_balcon_amount": intended_balcon_amount,
        "zip_count_4w": zip_count_4w,
        "velocity_6h": velocity_6h,
        "velocity_24h": velocity_24h,
        "velocity_4w": velocity_4w,
        "bank_branch_count_8w": bank_branch_count_8w,
        "date_of_birth_distinct_emails_4w": date_of_birth_distinct_emails_4w,
        "credit_risk_score": credit_risk_score,
        "email_is_free": email_is_free,
        "phone_home_valid": phone_home_valid,
        "phone_mobile_valid": phone_mobile_valid,
        "bank_months_count": bank_months_count,
        "has_other_cards": has_other_cards,
        "proposed_credit_limit": proposed_credit_limit,
        "foreign_request": foreign_request,
        "session_length_in_minutes": session_length_in_minutes,
        "keep_alive_session": keep_alive_session,
        "device_distinct_emails_8w": device_distinct_emails_8w,
        "device_fraud_count": device_fraud_count,
        "month": month,
        "address_diff": current_address_months_count - prev_address_months_count, 
        "housing_status": "dummy", 
        "device_os": "dummy",
        "source": "dummy",
        "payment_type": "dummy",
        "employment_status": "dummy"
    }
    user_df = pd.DataFrame([user_dict])
    
    # Agent decides
    score, decision, explanation = agent_decision(user_df)
    
    st.subheader("2. Fraud Risk Score")
    st.write(f"Predicted fraud probability: **{score:.2f}**")
    
    st.subheader("3. Agent Decision")
    st.write(f"Decision: **{decision}**")
    
    st.subheader("4. Explanation")
    st.write(explanation)
