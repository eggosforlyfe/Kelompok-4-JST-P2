import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from PIL import Image

# 1. Konfigurasi Halaman
st.set_page_config(
    page_title="AQI Pro-Predictor UNSRI", 
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Fungsi Load Assets (model JST)
@st.cache_resource
def load_assets():
    # Mengambil model, scaler, dan encoder yang sudah kamu simpan di Kaggle
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    le = pickle.load(open('label_encoder.pkl', 'rb'))
    return model, scaler, le

# Header Utama dengan gaya Dark Mode
st.title("🌌 Air Quality Intelligence - Beijing Dataset")
st.markdown("---")

# 3. Tab System untuk UI yang rapi
tab1, tab2 = st.tabs(["🔍 Prediksi Real-Time", "📊 Performa Model JST"])

with tab1:
    st.subheader("Input Parameter Lingkungan")
    
    # Input dibagi menjadi 3 kolom agar tidak terlalu panjang ke bawah
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 🌫️ Konsentrasi Polutan")
        pm10 = st.number_input("PM10 (µg/m³)", 0.0, 1000.0, 50.0)[cite: 1]
        so2 = st.number_input("SO2 (µg/m³)", 0.0, 500.0, 10.0)[cite: 1]
        no2 = st.number_input("NO2 (µg/m³)", 0.0, 500.0, 30.0)[cite: 1]
        co = st.number_input("CO (mg/m³)", 0.0, 10000.0, 800.0)[cite: 1]

    with col2:
        st.markdown("##### 🌡️ Kondisi Cuaca")
        o3 = st.number_input("O3 (µg/m³)", 0.0, 500.0, 50.0)[cite: 1]
        temp = st.slider("Suhu (TEMP) °C", -20.0, 50.0, 25.0)[cite: 1]
        pres = st.number_input("Tekanan (PRES) hPa", 900.0, 1100.0, 1010.0)[cite: 1]
        dewp = st.number_input("Titik Embun (DEWP) °C", -40.0, 40.0, 10.0)[cite: 1]

    with col3:
        st.markdown("##### 🌬️ Atmosfer")
        rain = st.number_input("Curah Hujan (RAIN) mm", 0.0, 100.0, 0.0)[cite: 1]
        wspm = st.number_input("Kecepatan Angin (WSPM)", 0.0, 50.0, 1.5)[cite: 1]
        wd_option = st.selectbox("Arah Angin (wd)", list(range(16)), help="Encoded Wind Direction 0-15")[cite: 1]
        
    st.markdown("---")
    
    if st.button("🚀 JALANKAN ANALISIS MLP"):
        model, scaler, le = load_assets()
        
        # Urutan fitur harus pas: ['PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','WSPM','wd'][cite: 1]
        data = np.array([[pm10, so2, no2, co, o3, temp, pres, dewp, rain, wspm, wd_option]])
        
        # Preprocessing & Prediksi[cite: 1]
        input_scaled = scaler.transform(data)
        prediction = model.predict(input_scaled)
        label = le.inverse_transform(prediction)[0]
        
        # Tampilan Hasil Prediksi
        st.subheader("Hasil Analisis:")
        if label == "Good":
            st.success(f"### Kategori: {label} ✅")
        elif label == "Moderate":
            st.warning(f"### Kategori: {label} ⚠️")
        elif label == "Unhealthy":
            st.error(f"### Kategori: {label} 😷")
        else:
            st.error(f"### Kategori: {label} 🚨 (BERBAHAYA)")

with tab2:
    st.subheader("Evaluasi Model JST (Multi-Layer Perceptron)")
    
    col_img, col_txt = st.columns([1, 1])
    
    with col_img:
        # Menampilkan Confusion Matrix yang kamu simpan dari Kaggle[cite: 1]
        if os.path.exists('confusion_matrix.png'):
            st.image('confusion_matrix.png', caption='Confusion Matrix Hasil Training')
        else:
            st.info("Visualisasi Confusion Matrix akan muncul di sini jika file 'confusion_matrix.png' sudah di-upload.")
            
    with col_txt:
        st.write("**Konfigurasi Arsitektur:**")
        # Detail arsitektur sesuai User Summary[cite: 1]
        st.code("""
        - Hidden Layers: (128, 64)
        - Activation: ReLU
        - Solver: Adam
        - Max Iterations: 1000
        - Dataset: Beijing Air Quality (3 Stations)
        """)
        st.markdown("""
        **Analisis Kelompok 4:**
        Model menunjukkan akurasi yang tinggi terutama pada kategori ekstrem. 
        Kategori 'Moderate' terkadang beririsan dengan 'Unhealthy' dikarenakan 
        distribusi data yang bersifat imbalance.[cite: 1]
        """)
