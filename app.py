import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------
st.set_page_config(page_title="Credit Default Risk Predictor", layout="centered")

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"  # rename to preprocessor.pkl if it's a
                                     # ColumnTransformer/Scaler instead of a
                                     # text vectorizer


# -----------------------------------------------------------------------
# LOAD ARTIFACTS
# -----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Missing `{MODEL_PATH}`. Drop your trained model file into the app folder.")
        st.stop()
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    vectorizer = None
    if os.path.exists(VECTORIZER_PATH):
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)

    return model, vectorizer


model, vectorizer = load_artifacts()

# -----------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------
st.title("💳 Credit Default Risk Predictor")
st.write(
    "Enter applicant details below to estimate the probability of loan default "
    "(TARGET = 1)."
)

with st.form("applicant_form"):
    col1, col2 = st.columns(2)

    with col1:
        amt_income_total = st.number_input("Annual Income (AMT_INCOME_TOTAL)", min_value=0.0, value=150000.0, step=1000.0)
        amt_credit = st.number_input("Credit Amount (AMT_CREDIT)", min_value=0.0, value=500000.0, step=1000.0)
        amt_annuity = st.number_input("Loan Annuity (AMT_ANNUITY)", min_value=0.0, value=25000.0, step=500.0)
        amt_goods_price = st.number_input("Goods Price (AMT_GOODS_PRICE)", min_value=0.0, value=450000.0, step=1000.0)
        years_birth = st.number_input("Age (years)", min_value=18, max_value=100, value=35)
        years_employed = st.number_input("Years Employed", min_value=0, max_value=60, value=5)

    with col2:
        name_income_type = st.selectbox(
            "Income Type",
            ["Working", "State servant", "Commercial associate", "Pensioner", "Student", "Unemployed"],
        )
        name_education_type = st.selectbox(
            "Education",
            ["Secondary / secondary special", "Higher education", "Incomplete higher",
             "Lower secondary", "Academic degree"],
        )
        name_family_status = st.selectbox(
            "Family Status",
            ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
        )
        occupation_type = st.selectbox(
            "Occupation Type",
            ["Laborers", "Sales staff", "Core staff", "Managers", "Drivers",
             "High skill tech staff", "Accountants", "Others"],
        )
        cnt_fam_members = st.number_input("Family Members Count", min_value=1, max_value=20, value=2)
        code_gender = st.selectbox("Gender", ["M", "F"])

    submitted = st.form_submit_button("Predict")

# -----------------------------------------------------------------------
# BUILD FEATURE FRAME
# -----------------------------------------------------------------------
def build_input_frame():
    """
    Adjust this dict so its keys/values match EXACTLY the feature set and
    encoding your trained model/preprocessor expects (same column names,
    same order, same categorical codes used during training).
    """
    data = {
        "AMT_INCOME_TOTAL": [amt_income_total],
        "AMT_CREDIT": [amt_credit],
        "AMT_ANNUITY": [amt_annuity],
        "AMT_GOODS_PRICE": [amt_goods_price],
        "YEARS_BIRTH": [years_birth],
        "YEARS_EMPLOYED": [years_employed],
        "NAME_INCOME_TYPE": [name_income_type],
        "NAME_EDUCATION_TYPE": [name_education_type],
        "NAME_FAMILY_STATUS": [name_family_status],
        "OCCUPATION_TYPE": [occupation_type],
        "CNT_FAM_MEMBERS": [cnt_fam_members],
        "CODE_GENDER": [code_gender],
    }
    return pd.DataFrame(data)


if submitted:
    input_df = build_input_frame()

    st.subheader("Input Summary")
    st.dataframe(input_df)

    try:
        # If you used a ColumnTransformer/encoder saved as vectorizer.pkl,
        # transform the raw input before predicting.
        X = vectorizer.transform(input_df) if vectorizer is not None else input_df

        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else None

        st.subheader("Prediction")
        if pred == 1:
            st.error(f"⚠️ High risk of default (TARGET = 1)")
        else:
            st.success(f"✅ Low risk of default (TARGET = 0)")

        if proba is not None:
            st.metric("Default Probability", f"{proba * 100:.2f}%")
            st.progress(min(max(proba, 0.0), 1.0))

    except Exception as e:
        st.error(
            "Prediction failed. This usually means the input columns don't match "
            "what the model/vectorizer was trained on. Update `build_input_frame()` "
            "in app.py to match your training feature schema.\n\n"
            f"Error: {e}"
        )

st.markdown("---")
st.caption(
    "This app expects `model.pkl` (and optionally `vectorizer.pkl`/`preprocessor.pkl`) "
    "in the same directory. Replace the placeholder feature list in `build_input_frame()` "
    "with the exact columns your model was trained on."
)
