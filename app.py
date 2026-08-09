import streamlit as st
import pandas as pd
import joblib


# Load model and feature columns

model = joblib.load("insurance_charge_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")


# Page Configuration

st.set_page_config(
    page_title="Insurance Charge Predictor",
    page_icon="🏥",
    layout="centered"
)

# Title

st.title("🏥 Insurance Charge Predictor")

st.write(
    "Enter customer information to estimate the expected insurance charge."
)

st.caption(
    "Educational machine learning application — not financial or insurance advice."
)


# Customer Information

st.subheader("Customer Information")

age = st.slider("Age",min_value=18,max_value=64,value=30)

sex = st.selectbox( "Gender",["Female", "Male"]
)

bmi = st.number_input("BMI",min_value=10.0,max_value=60.0,value=25.0,step=0.1)

children = st.number_input("Number of Children", min_value=0, max_value=7, value=0, step=1)

smoker = st.selectbox( "Smoker", ["No", "Yes"])

region = st.selectbox("Region",[ "northeast", "northwest", "southeast", "southwest"])


# Prediction

if st.button("💰 Predict Insurance Charge"):

    # Create input data
    input_data = pd.DataFrame({
        "age": [age],

        "isfemale": [1 if sex == "Female" else 0],

        "bmi": [bmi],

        "children": [children],

        "issmoker": [ 1 if smoker == "Yes" else 0],

        "region_northeast": [1 if region == "northeast" else 0],

        "region_northwest": [1 if region == "northwest" else 0],

        "region_southeast": [1 if region == "southeast" else 0],

        "region_southwest": [1 if region == "southwest" else 0]
    })


    # Ensure same column order as training
    input_data = input_data[feature_columns]


    # Prediction
    prediction = model.predict(input_data)[0]


    # Display result
    st.success(
        f"### Estimated Insurance Charge: ${prediction:,.2f}"
    )