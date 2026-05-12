import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="AQI Predictor - JST Beijing",
    page_icon="🌍",
    layout="wide"
)

# --- CSS CUSTOM ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL & SCALER ---
@st.cache_resource
def load_assets():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except:
    st.error("⚠️ File model.pkl atau scaler.pkl tidak ditemukan!")

# --- SIDEBAR NAVIGASI ---
with st.sidebar:
    st.title("Navigasi")
    page = st.radio("Pilih Menu:", ["Prediksi Real-time", "Performa Model JST"])
    st.divider()
    st.info("Aplikasi JST untuk klasifikasi AQI Beijing.")

# --- HALAMAN 1: PREDIKSI REAL-TIME ---
if page == "Prediksi Real-time":
    st.title("🌍 Prediksi Kualitas Udara Real-time")
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🧪 Parameter Polutan")
            pm10 = st.number_input("PM10", value=0.0)
            so2 = st.number_input("SO2", value=0.0)
            no2 = st.number_input("NO2", value=0.0)
            co = st.number_input("CO", value=0.0)
            o3 = st.number_input("O3", value=0.0)
        
        with col2:
            st.subheader("🌡️ Parameter Meteorologi")
            temp = st.number_input("Suhu (°C)", value=0.0)
            pres = st.number_input("Tekanan Udara (hPa)", value=0.0)
            dewp = st.number_input("Titik Embun (°C)", value=0.0)
            rain = st.number_input("Curah Hujan (mm)", value=0.0)
            wspm = st.number_input("Kec. Angin (m/s)", value=0.0)
            wd = st.selectbox("Arah Angin", ['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW', 'NNW', 'NNE', 'SSW', 'SSE', 'WNW', 'WSW', 'ESE', 'ENE'])

    wd_map = {'E': 0, 'ENE': 1, 'ESE': 2, 'N': 3, 'NE': 4, 'NNE': 5, 'NNW': 6, 'NW': 7, 
              'S': 8, 'SE': 9, 'SSE': 10, 'SSW': 11, 'SW': 12, 'W': 13, 'WNW': 14, 'WSW': 15}
    wd_encoded = wd_map[wd]

    if st.button("🚀 Analisis Kualitas Udara"):
        features = np.array([[pm10, so2, no2, co, o3, temp, pres, dewp, rain, wspm, wd_encoded]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0]

        aqi_info = {
            0: {"label": "Good", "color": "#28a745", "desc": "Kualitas udara sangat baik. Tidak ada risiko kesehatan."},
            1: {"label": "Moderate", "color": "#ffc107", "desc": "Kualitas udara dapat diterima. Kelompok sensitif perlu waspada."},
            2: {"label": "Unhealthy", "color": "#fd7e14", "desc": "Kualitas udara mulai merugikan kesehatan masyarakat."},
            3: {"label": "Hazardous", "color": "#dc3545", "desc": "Kondisi darurat! Udara sangat berbahaya bagi semua orang."}
        }
        res = aqi_info[prediction]

        st.divider()
        st.markdown(f"""
            <div style="background-color:{res['color']}; padding:30px; border-radius:15px; text-align:center;">
                <h1 style="color:white; margin:0;">{res['label'].upper()}</h1>
                <p style="color:white; font-size:18px;">{res['desc']}</p>
            </div>
        """, unsafe_allow_html=True)

        with st.expander("🔍 Bagaimana hasil ini didapatkan?"):
            st.write(f"Model JST menganalisis **11 fitur** yang kamu masukkan. Berdasarkan pembobotan (weight) yang dipelajari di Kaggle, input kamu memiliki kecocokan **{prob[prediction]*100:.2f}%** dengan pola data historis kategori **{res['label']}**.")
            st.write("Indikator dominan dalam prediksi ini biasanya dipengaruhi oleh kadar polutan gas (CO/PM10) dan kecepatan angin (WSPM).")
            
# --- HALAMAN 2: PERFORMA MODEL JST ---
else:
    st.title("📈 Performa Model JST")

    st.divider()

    # --- TAMBAHAN VISUALISASI ---
    st.subheader("🖼️ Visualisasi Analisis & Evaluasi")
    
    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Distribusi AQI", "Heatmap Korelasi"])
    
    with tab1:
        st.write("Menunjukkan seberapa akurat model menebak tiap kelas AQI.")
        try:
            st.image("confusion_matrix.png", use_container_width=True)
        except:
            st.warning("Foto 'confusion_matrix.png' tidak ditemukan.")

    with tab2:
        st.write("Menunjukkan perbandingan jumlah data untuk setiap kategori kualitas udara (EDA).")
        try:
            st.image("distribusi_aqi.png", use_container_width=True)
        except:
            st.warning("Foto 'distribusi_aqi.png' tidak ditemukan.")

    with tab3:
        st.write("Menunjukkan hubungan keterkaitan antara parameter lingkungan (EDA).")
        try:
            st.image("heatmap_korelasi.png", use_container_width=True)
        except:
            st.warning("Foto 'heatmap_korelasi.png' tidak ditemukan.")

    st.subheader("🧠 Arsitektur JST")
    st.json({
        "Model": "Multi-Layer Perceptron",
        "Activation": "ReLU",
        "Optimizer": "Adam",
        "Preprocessing": "Quantile Transformation"
    })
