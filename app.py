import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="Loan Prediction System",
    page_icon="💳",
    layout="wide"
)

# ------------------------------
# CUSTOM CSS
# ------------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fb;
    }

    .title {
        font-size: 48px;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 10px;
    }

    .subtitle {
        font-size: 18px;
        color: #4B5563;
        text-align: center;
        margin-bottom: 30px;
    }

    .card {
        background-color: white;
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    }

    .metric-card {
        background: linear-gradient(135deg, #2563EB, #1E40AF);
        padding: 20px;
        border-radius: 18px;
        color: white;
        text-align: center;
    }

    .prediction-success {
        background-color: #DCFCE7;
        color: #166534;
        padding: 20px;
        border-radius: 15px;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
    }

    .prediction-fail {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 20px;
        border-radius: 15px;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
    }

    .stButton>button {
        background: linear-gradient(135deg, #2563EB, #1E40AF);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 25px;
        font-size: 18px;
        font-weight: bold;
        width: 100%;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #1D4ED8, #1E3A8A);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------
# LOAD MODEL
# ------------------------------

# Replace with your actual saved model file
model = pickle.load(open('loan_model.pkl', 'rb'))

# ------------------------------
# HEADER
# ------------------------------

st.markdown("<div class='title'>💳 Loan Prediction System</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Predict whether a loan application will be Approved or Rejected using Machine Learning</div>",
    unsafe_allow_html=True
)

# ------------------------------
# SIDEBAR
# ------------------------------

st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📊 Predict Loan", "📈 Dataset Insights", "👩‍💻 About Project"]
)

# ------------------------------
# HOME PAGE
# ------------------------------

if page == "🏠 Home":

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class='metric-card'>
            <h2>🤖 Model</h2>
            <h3>Logistic Regression</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class='metric-card'>
            <h2>📂 Dataset</h2>
            <h3>Kaggle Loan Dataset</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class='metric-card'>
            <h2>⚡ ML Task</h2>
            <h3>Binary Classification</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    st.markdown("""
    ### 🔍 Project Overview

    This application predicts whether a customer is eligible for a loan approval based on multiple factors such as:

    - Applicant Income
    - Coapplicant Income
    - Loan Amount
    - Credit History
    - Property Area
    - Education
    - Marital Status
    - Employment Status

    The model used is **Logistic Regression**, trained after performing data preprocessing and feature engineering.
    """)

    st.info("👉 Navigate to the 'Predict Loan' section from the sidebar to test the model.")

# ------------------------------
# PREDICTION PAGE
# ------------------------------

elif page == "📊 Predict Loan":

    st.markdown("## 📝 Enter Applicant Details")

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", [0, 1, 2, 3])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
            applicant_income = st.number_input("Applicant Income", min_value=0)

        with col2:
            coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
            loan_amount = st.number_input("Loan Amount", min_value=0)
            loan_amount_term = st.number_input("Loan Amount Term", min_value=0)
            credit_history = st.selectbox("Credit History", [1.0, 0.0])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        st.markdown("</div>", unsafe_allow_html=True)

    # ------------------------------
    # ENCODING INPUTS
    # ------------------------------

    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0

    if property_area == "Urban":
        property_area = 2
    elif property_area == "Semiurban":
        property_area = 1
    else:
        property_area = 0

    input_data = np.array([
        gender,
        married,
        dependents,
        education,
        self_employed,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_amount_term,
        credit_history,
        property_area
    ]).reshape(1, -1)

    # ------------------------------
    # PREDICTION
    # ------------------------------

    if st.button("🚀 Predict Loan Status"):

        prediction = model.predict(input_data)

        st.write("")

        if prediction[0] == 1:
            st.markdown(
                """
                <div class='prediction-success'>
                ✅ Loan Approved
                </div>
                """,
                unsafe_allow_html=True
            )
            st.balloons()

        else:
            st.markdown(
                """
                <div class='prediction-fail'>
                ❌ Loan Rejected
                </div>
                """,
                unsafe_allow_html=True
            )

# ------------------------------
# DATASET INSIGHTS
# ------------------------------

elif page == "📈 Dataset Insights":

    st.markdown("## 📊 Dataset Insights")

    try:
        df = pd.read_csv("Loan Prediction Train.csv")

        st.markdown("### 🔹 Dataset Preview")
        st.dataframe(df.head())

        st.markdown("### 🔹 Dataset Shape")
        st.write(f"Rows: {df.shape[0]}")
        st.write(f"Columns: {df.shape[1]}")

        st.markdown("### 🔹 Missing Values")
        st.dataframe(df.isnull().sum())

        st.markdown("### 🔹 Statistical Summary")
        st.dataframe(df.describe())

        st.markdown("### 🔹 Loan Status Distribution")
        st.bar_chart(df['Loan_Status'].value_counts())

    except:
        st.warning("Dataset file not found. Add 'Loan Prediction Train.csv' to your project folder.")

# ------------------------------
# ABOUT PAGE
# ------------------------------

elif page == "👩‍💻 About Project":

    st.markdown("## 👩‍💻 About This Project")

    st.markdown(
        """
        ### 📌 Project Details

        **Project Name:** Loan Prediction Using Machine Learning

        **Dataset Source:** Kaggle Loan Prediction Dataset

        **Machine Learning Algorithm:** Logistic Regression

        ### 🛠️ Technologies Used

        - Python
        - Pandas
        - NumPy
        - Scikit-learn
        - Streamlit

        ### ⚙️ Workflow

        1. Data Collection
        2. Data Cleaning
        3. Data Preprocessing
        4. Feature Encoding
        5. Model Training
        6. Model Evaluation
        7. Streamlit Deployment

        ### 🎯 Objective

        To automate loan approval prediction using applicant details and machine learning.
        """
    )

    st.success("✨ Streamlit app designed for portfolio projects, mini projects, and academic presentations.")

# ------------------------------
# FOOTER
# ------------------------------

st.write("")
st.markdown("---")
st.caption("Made by Gagandeep Kaur using Streamlit and Machine Learning")
