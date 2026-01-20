import streamlit as st
import pandas as pd
from agent.fraud_agent import agent_decision, get_feature_names

st.title("Fraud Detection Agent")

st.header("1. Enter Application Details")

with st.form("transaction_form"):
    income = st.slider("Income (quantile)", 0.0, 1.0, 0.5)
    name_email_similarity = st.slider("Name–Email Similarity", 0.0, 1.0, 0.5)

    prev_address_months_count = st.number_input(
        "Previous Address Months (-1 if missing)", min_value=-1, max_value=380, value=12
    )

    current_address_months_count = st.number_input(
        "Current Address Months (-1 if missing)", min_value=-1, max_value=406, value=24
    )

    customer_age = st.selectbox(
        "Customer Age Bucket", [20, 30, 40, 50, 60]
    )

    days_since_request = st.slider(
        "Days Since Request", 0, 78, 10
    )

    intended_balcon_amount = st.number_input(
        "Intended Balance Amount", min_value=-1.0, max_value=108.0, value=20.0
    )

    zip_count_4w = st.number_input(
        "ZIP Count (last 4 weeks)", min_value=1, max_value=5767, value=10
    )
    submitted = st.form_submit_button("Compute Decision")

if submitted:
    user_dict = {
        "income": income,
        "name_email_similarity": name_email_similarity,
        "prev_address_months_count": prev_address_months_count,
        "current_address_months_count": current_address_months_count,
        "customer_age": customer_age,
        "days_since_request": days_since_request,
        "intended_balcon_amount": intended_balcon_amount,
        "zip_count_4w": zip_count_4w,
        "address_diff": current_address_months_count - prev_address_months_count
    }
    user_df = pd.DataFrame([user_dict])

    st.write("Input DF:", user_df)
    st.write("Dtypes:", user_df.dtypes)

    st.write("Raw probability:", get_feature_names(user_df))

    
    # Agent decides
    score, decision, explanation = agent_decision(user_df)
    
    st.subheader("2. Fraud Risk Score")
    st.write(f"Predicted fraud probability: **{score:.4f}**")
    
    st.subheader("3. Agent Decision")
    st.write(f"Decision: **{decision}**")
    
    st.subheader("4. Explanation")
    st.write(explanation)
