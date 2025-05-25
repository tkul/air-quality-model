import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="PM2.5 Prediction", layout="centered")

st.title("🌍 PM2.5 Air Pollution Prediction App")
st.write("Estimate PM2.5 levels based on environmental and regional input data.")

@st.cache_resource
def load_model():
    model = joblib.load("xgb_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_model()

st.subheader("Input Parameters")

with st.form("prediction_form"):
    pm10 = st.slider("PM10 (μg/m³)", 0.0, 200.0, 50.0)
    no2 = st.slider("NO2 (μg/m³)", 0.0, 200.0, 30.0)
    pm10_cov = st.slider("PM10 Temporal Coverage (%)", 0.0, 100.0, 90.0)
    no2_cov = st.slider("NO2 Temporal Coverage (%)", 0.0, 100.0, 90.0)
    pm25_cov = st.slider("PM2.5 Temporal Coverage (%)", 0.0, 100.0, 90.0)

    who_region = st.selectbox("WHO Region", [
        "African Region", "Region of the Americas", "South-East Asia Region", 
        "European Region", "Eastern Mediterranean Region", "Western Pacific Region"
    ])

    iso3 = st.text_input("ISO3 Country Code", "TUR")
    country = st.text_input("Country Name", "Turkey")

    submit = st.form_submit_button("Predict")

if submit:
    input_df = pd.DataFrame([{
        "PM10 (μg/m3)": pm10,
        "NO2 (μg/m3)": no2,
        "PM10 temporal coverage (%)": pm10_cov,
        "NO2 temporal coverage (%)": no2_cov,
        "PM25 temporal coverage (%)": pm25_cov,
        "WHO Region": who_region,
        "ISO3": iso3,
        "WHO Country Name": country
    }])

    try:
        transformed = preprocessor.transform(input_df)
        prediction = model.predict(transformed)[0]
        st.success(f"Estimated PM2.5 Level: {prediction:.2f} μg/m³")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
