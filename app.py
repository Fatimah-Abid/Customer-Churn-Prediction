import streamlit as st
import pandas as pd
import joblib

from google import genai


# ==============================
# Load Model and Preprocessor
# ==============================

model = joblib.load("churn_xgboost_model.pkl")
preprocessor = joblib.load("churn_preprocessor.pkl")


# ==============================
# Gemini API
# Streamlit Cloud Secrets
# ==============================

try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    gemini_api_key = None

if gemini_api_key:
    client = genai.Client(api_key=gemini_api_key)
else:
    client = None


# ==============================
# Page Configuration
# ==============================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ==============================
# App Title
# ==============================

st.title("📊 Customer Churn Prediction")

st.write(
    "Enter customer information to predict churn probability "
    "and understand the customer's risk."
)


# ==============================
# Customer Information
# ==============================

st.header("👤 Customer Information")

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    tenure = st.number_input(
        "Tenure",
        min_value=1,
        value=12
    )


with col2:

    subscription_type = st.selectbox(
        "Subscription Type",
        ["Basic", "Standard", "Premium"]
    )

    contract_length = st.selectbox(
        "Contract Length",
        ["Monthly", "Quarterly", "Annual"]
    )


# ==============================
# Usage Information
# ==============================

st.header("📈 Usage Information")

col1, col2 = st.columns(2)

with col1:

    usage_frequency = st.number_input(
        "Usage Frequency",
        min_value=0,
        value=10
    )

    support_calls = st.number_input(
        "Support Calls",
        min_value=0,
        value=2
    )


with col2:

    last_interaction = st.number_input(
        "Last Interaction",
        min_value=1,
        value=10
    )


# ==============================
# Payment Information
# ==============================

st.header("💳 Payment Information")

col1, col2 = st.columns(2)

with col1:

    payment_delay = st.number_input(
        "Payment Delay",
        min_value=0,
        value=5
    )

with col2:

    total_spend = st.number_input(
        "Total Spend",
        min_value=0.0,
        value=500.0
    )


# ==============================
# Prediction
# ==============================

if st.button(
    "Predict Churn",
    use_container_width=True
):

    # ==============================
    # Feature Engineering
    # ==============================

    total_activity = (
        usage_frequency + support_calls
    )

    support_calls_per_tenure = (
        support_calls / (tenure + 1)
    )

    payment_delay_per_tenure = (
        payment_delay / (tenure + 1)
    )


    # ==============================
    # Customer Data
    # ==============================

    customer_data = pd.DataFrame({

        "Age": [age],

        "Gender": [gender],

        "Tenure": [tenure],

        "Usage Frequency": [
            usage_frequency
        ],

        "Support Calls": [
            support_calls
        ],

        "Payment Delay": [
            payment_delay
        ],

        "Subscription Type": [
            subscription_type
        ],

        "Contract Length": [
            contract_length
        ],

        "Total Spend": [
            total_spend
        ],

        "Last Interaction": [
            last_interaction
        ],

        "Total_Activity": [
            total_activity
        ],

        "Support_Calls_per_Tenure": [
            support_calls_per_tenure
        ],

        "Payment_Delay_per_Tenure": [
            payment_delay_per_tenure
        ]
    })


    # ==============================
    # Preprocessing
    # ==============================

    customer_processed = preprocessor.transform(
        customer_data
    )


    # ==============================
    # Prediction
    # ==============================

    prediction = model.predict(
        customer_processed
    )[0]

    probability = model.predict_proba(
        customer_processed
    )[0][1]


    # ==============================
    # Risk Segment
    # ==============================

    if probability < 0.3:

        segment = "Low"

    elif probability <= 0.7:

        segment = "Medium"

    else:

        segment = "High"


    # ==============================
    # Prediction Result
    # ==============================

    st.divider()

    st.header("📊 Prediction Result")

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )


    with col2:

        st.metric(
            "Risk Segment",
            segment
        )


    with col3:

        if prediction == 1:

            st.metric(
                "Prediction",
                "Likely to Churn"
            )

        else:

            st.metric(
                "Prediction",
                "Unlikely to Churn"
            )


    # ==============================
    # Risk Message
    # ==============================

    if segment == "High":

        st.error(
            "⚠️ High churn risk. "
            "Immediate retention attention may be required."
        )

    elif segment == "Medium":

        st.warning(
            "⚠️ Medium churn risk. "
            "The customer should be monitored."
        )

    else:

        st.success(
            "✅ Low churn risk. "
            "The customer currently appears stable."
        )


    # ==============================
    # Customer Summary
    # ==============================

    st.subheader("🔎 Customer Summary")

    st.write(f"**Age:** {age}")
    st.write(f"**Tenure:** {tenure}")
    st.write(f"**Usage Frequency:** {usage_frequency}")
    st.write(f"**Support Calls:** {support_calls}")
    st.write(f"**Payment Delay:** {payment_delay}")
    st.write(f"**Contract:** {contract_length}")
    st.write(f"**Total Spend:** {total_spend}")


    # ==============================
    # Gemini AI Explanation
    # ==============================

    st.divider()

    st.header("🤖 AI-Powered Explanation")


    if client is None:

        st.warning(
            "Gemini API key is not configured."
        )

    else:

        # ==============================
        # Safe Customer Summary
        # ==============================

        safe_summary = f"""
Churn probability: {probability:.2%}

Risk segment: {segment}

Prediction: {
    "Likely to churn"
    if prediction == 1
    else "Unlikely to churn"
}

Customer features:

Age: {age}
Tenure: {tenure}
Usage Frequency: {usage_frequency}
Support Calls: {support_calls}
Payment Delay: {payment_delay}
Subscription Type: {subscription_type}
Contract Length: {contract_length}
Total Spend: {total_spend}
Last Interaction: {last_interaction}
"""


        # ==============================
        # Gemini Prompt
        # ==============================

        prompt = f"""
You are a customer retention analyst.

Analyze the following customer churn prediction.

Use ONLY the information provided below.
Do not invent information.

Explain the result in simple, professional language.

Include:

1. Why the customer has this risk level.
2. The 2 or 3 most important factors visible in the provided data.
3. Two practical customer-retention suggestions.

Keep the explanation concise.

Customer Summary:

{safe_summary}
"""


        # ==============================
        # Gemini Response
        # ==============================

        try:

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            st.write(response.text)

        except Exception as e:

            st.warning(
                "Gemini explanation could not be generated."
            )

            st.caption(
                f"Error: {e}"
            )