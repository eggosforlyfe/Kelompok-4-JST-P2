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
    # Pastikan file ini ada di folder yang sama
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except:
    st.error("⚠️ File model.pkl atau scaler.pkl tidak ditemukan. Pastikan sudah di-upload!")

# --- SIDEBAR NAVIGASI ---
with st.sidebar:
    st.image("https://www.gstatic.com/devrel-devsite/prod/v2400b84c8d5047b8538b8120092d6e3f2824c65a04a39f603c4f74d9e03d3c8c/tensorflow/images/lockup.png", width=200)
    st.title("Navigasi")
    page = st.radio("Pilih Menu:", ["Prediksi Real-time", "Performa Model JST"])
    st.divider()
    st.info("Aplikasi ini menggunakan Jaringan Syaraf Tiruan (MLP) untuk klasifikasi AQI Beijing.")

# --- HALAMAN 1: PREDIKSI REAL-TIME ---
if page == "Prediksi Real-time":
    st.title("🌍 Prediksi Kualitas Udara Real-time")
    st.write("Masukkan parameter di bawah untuk mengetahui kategori kualitas udara.")

    # Form Input
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

    # Map Arah Angin (Sesuaikan dengan LabelEncoder di Kaggle)
    wd_map = {'E': 0, 'ENE': 1, 'ESE': 2, 'N': 3, 'NE': 4, 'NNE': 5, 'NNW': 6, 'NW': 7, 
              'S': 8, 'SE': 9, 'SSE': 10, 'SSW': 11, 'SW': 12, 'W': 13, 'WNW': 14, 'WSW': 15}
    wd_encoded = wd_map[wd]

    if st.button("🚀 Analisis Kualitas Udara"):
        # Proses Prediksi
        features = np.array([[pm10, so2, no2, co, o3, temp, pres, dewp, rain, wspm, wd_encoded]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0]

        # Definisi Hasil
        aqi_info = {
            0: {"label": "Good", "color": "#28a745", "desc": "Kualitas udara sangat baik. Tidak ada risiko kesehatan bagi masyarakat."},
            1: {"label": "Moderate", "color": "#ffc107", "desc": "Kualitas udara dapat diterima. Namun, bagi kelompok sensitif perlu waspada."},
            2: {"label": "Unhealthy", "color": "#fd7e14", "desc": "Kualitas udara mulai merugikan kesehatan. Masyarakat umum disarankan mengurangi aktivitas luar ruangan."},
            3: {"label": "Hazardous", "color": "#dc3545", "desc": "Kondisi darurat kesehatan! Udara sangat beracun dan berbahaya bagi semua orang."}
        }

        res = aqi_info[prediction]

        # Tampilan Hasil
        st.divider()
        st.markdown(f"""
            <div style="background-color:{res['color']}; padding:30px; border-radius:15px; text-align:center;">
                <h1 style="color:white; margin:0;">{res['label'].upper()}</h1>
                <p style="color:white; font-size:18px;">{res['desc']}</p>
            </div>
        """, unsafe_allow_html=True)

        # Penjelasan "Didapatkan dari mana?"
        with st.expander("🔍 Bagaimana hasil ini didapatkan?"):
            st.write(f"Model JST menganalisis **11 fitur** yang kamu masukkan. Berdasarkan pembobotan (weight) yang dipelajari di Kaggle, input kamu memiliki kecocokan **{prob[prediction]*100:.2f}%** dengan pola data historis kategori **{res['label']}**.")
            st.write("Indikator dominan dalam prediksi ini biasanya dipengaruhi oleh kadar polutan gas (CO/PM10) dan kecepatan angin (WSPM).")

# --- HALAMAN 2: PERFORMA MODEL JST ---
else:
    st.title("📈 Performa Model JST")
    st.write("Detail teknis hasil fine-tuning model untuk mencapai akurasi 90%+")

    col1, col2, col3 = st.columns(3)
    col1.metric("Akurasi Final", "92.45%") # Sesuaikan dengan angka hasil trainingmu
    col2.metric("Loss Terendah", "0.084")
    col3.metric("Epochs", "1000")

    st.subheader("🧠 Arsitektur Fine-Tuning")
    st.json({
        "Input Layer": "11 Fitur (Polutan & Meteorologi)",
        "Hidden Layers": "3 Layer (512, 512, 256)",
        "Activation Function": "ReLU",
        "Preprocessing": "Quantile Transformation",
        "Optimizer": "Adam",
        "Regularization": "Alpha 0.0001"
    })

    st.subheader("📊 Evaluasi Model")
    st.write("Visualisasi Confusion Matrix dan Classification Report dari hasil training Kaggle.")
    # Kamu bisa upload gambar Confusion Matrix ke folder yang sama dan panggil di sini
    # st.image("confusion_matrix.png", caption="Confusion Matrix - Hasil Akurasi 90%+")
    st.info("Model berhasil meminimalkan salah prediksi pada kategori 'Hazardous' berkat teknik Balancing dan Fine-tuning neuron.")
