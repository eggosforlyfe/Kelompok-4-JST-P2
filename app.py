import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from PIL import Image

# 1. Konfigurasi Halaman & Tema Gelap via CSS
st.set_page_config(page_title="AQI Pro-Predictor", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .stNumberInput>div>div>input {
        background-color: #262730;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Fungsi Load Assets
@st.cache_resource
def load_assets():
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    le = pickle.load(open('label_encoder.pkl', 'rb'))
    return model, scaler, le

# Header Utama
st.title("🌌 Air Quality Intelligence - Beijing")
st.markdown("---")

# 3. Layout Utama (Tab)
tab1, tab2 = st.tabs(["🔍 Prediksi Real-Time", "📊 Performa Model JST"])

with tab1:
    st.subheader("Input Parameter Lingkungan")
    
    # Membagi input menjadi 3 kolom agar rapi
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 🌫️ Partikulat & Gas")
        pm10 = st.number_input("PM10 (µg/m³)", 0.0, 1000.0, 50.0)
        so2 = st.number_input("SO2 (µg/m³)", 0.0, 500.0, 10.0)
        no2 = st.number_input("NO2 (µg/m³)", 0.0, 500.0, 30.0)
        co = st.number_input("CO (mg/m³)", 0.0, 10000.0, 800.0)

    with col2:
        st.markdown("##### 🌡️ Cuaca")
        o3 = st.number_input("O3 (µg/m³)", 0.0, 500.0, 50.0)
        temp = st.slider("Suhu (TEMP) °C", -20.0, 50.0, 25.0)
        pres = st.number_input("Tekanan (PRES) hPa", 900.0, 1100.0, 1010.0)
        dewp = st.number_input("Titik Embun (DEWP) °C", -40.0, 40.0, 10.0)

    with col3:
        st.markdown("##### 🌬️ Angin & Hujan")
        rain = st.number_input("Curah Hujan (RAIN) mm", 0.0, 100.0, 0.0)
        wspm = st.number_input("Kecepatan Angin (WSPM)", 0.0, 50.0, 1.5)
        wd_option = st.selectbox("Arah Angin (wd)", list(range(16)), 
                                help="0: N, 1: NNE, 2: NE, ... dst")
        
    st.markdown("---")
    
    if st.button("🚀 ANALISIS KUALITAS UDARA"):
        model, scaler, le = load_assets()
        
        # Susun data sesuai urutan fitur Kaggle
        data = np.array([[pm10, so2, no2, co, o3, temp, pres, dewp, rain, wspm, wd_option]])
        
        # Scaling & Prediksi
        input_scaled = scaler.transform(data)
        prediction = model.predict(input_scaled)
        label = le.inverse_transform(prediction)[0]
        
        # Tampilan Hasil yang Keren
        st.subheader("Hasil Analisis:")
        if label == "Good":
            st.balloons()
            st.success(f"### Kualitas Udara: **{label}** (Sangat Baik)")
        elif label == "Moderate":
            st.warning(f"### Kualitas Udara: **{label}** (Sedang/Biasa)")
        elif label == "Unhealthy":
            st.error(f"### Kualitas Udara: **{label}** (Tidak Sehat)")
        else:
            st.critical(f"### Kualitas Udara: **{label}** (BERBAHAYA)")

with tab2:
    st.subheader("Statistik Performa Neural Network (MLP)")
    
    col_a, col_b = st.columns([1, 1])
    
    with col_a:
        st.markdown("**Confusion Matrix**")
        if os.path.exists('confusion_matrix.png'):
            image = Image.open('confusion_matrix.png')
            st.image(image, caption='Akurasi Model pada Data Testing', use_container_width=True)
        else:
            st.info("Upload 'confusion_matrix.png' ke GitHub untuk melihat visualisasi.")
            
    with col_b:
        st.markdown("**Detail Model**")
        st.json({
            "Algorithm": "Multi-Layer Perceptron (MLP)",
            "Architecture": "128 Hidden Layer 1, 64 Hidden Layer 2",
            "Activation": "ReLU",
            "Optimization": "Adam",
            "Max Iterations": 1000
        })
        st.markdown("""
            *   **Akurasi**: Model memiliki kemampuan klasifikasi yang baik pada kelas *Hazardous* dan *Unhealthy*.
            *   **Data**: Menggunakan 3 stasiun terbaik (Aotizhongxin, Changping, Dongsi)[cite: 1].
        """)

st.sidebar.markdown("---")
st.sidebar.write("Proyek JST Kelompok 4")
