import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Load semua komponen yang sudah disimpan
def load_assets():
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    le = pickle.load(open('label_encoder.pkl', 'rb'))
    return model, scaler, le

model, scaler, le = load_assets()

# 2. UI App
st.set_page_config(page_title="AQI Classifier - Beijing", layout="wide")
st.title("🌱 Klasifikasi Kualitas Udara (Beijing Dataset)")
st.write("Aplikasi ini memprediksi kategori AQI menggunakan Multi-Layer Perceptron.")

st.sidebar.header("Input Parameter Cuaca")

# Form Input sesuai fitur: ['PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','WSPM','wd']
def user_input_features():
    pm10 = st.sidebar.number_input("PM10", value=0.0)
    so2 = st.sidebar.number_input("SO2", value=0.0)
    no2 = st.sidebar.number_input("NO2", value=0.0)
    co = st.sidebar.number_input("CO", value=0.0)
    o3 = st.sidebar.number_input("O3", value=0.0)
    temp = st.sidebar.slider("Temperature (TEMP)", -20.0, 45.0, 15.0)
    pres = st.sidebar.number_input("Pressure (PRES)", value=1010.0)
    dewp = st.sidebar.number_input("Dew Point (DEWP)", value=0.0)
    rain = st.sidebar.number_input("Rainfall", value=0.0)
    wspm = st.sidebar.number_input("Wind Speed (WSPM)", value=0.0)
    wd = st.sidebar.selectbox("Wind Direction (Encoded)", list(range(16)))
    
    data = {
        'PM10': pm10, 'SO2': so2, 'NO2': no2, 'CO': co, 'O3': o3,
        'TEMP': temp, 'PRES': pres, 'DEWP': dewp, 'RAIN': rain,
        'WSPM': wspm, 'wd': wd
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

st.subheader("Data Input User")
st.write(input_df)

if st.button("Prediksi"):
    # Preprocessing: Scaling
    input_scaled = scaler.transform(input_df)
    
    # Prediksi
    prediction = model.predict(input_scaled)
    prediction_label = le.inverse_transform(prediction)
    
    # Hasil
    st.subheader("Hasil Klasifikasi")
    color = "green" if prediction_label[0] == "Good" else "orange" if prediction_label[0] == "Moderate" else "red"
    st.markdown(f"### Kualitas Udara: :{color}[{prediction_label[0]}]")
