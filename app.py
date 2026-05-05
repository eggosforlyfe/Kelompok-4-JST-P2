import streamlit as st
import os

st.title("Debug Mode - AQI UNSRI")

# Cek apakah file benar-benar ada di server
st.write("Daftar file di server saat ini:")
st.write(os.listdir("."))

# Cek apakah file model terbaca
if os.path.exists("model.pkl"):
    st.success("File model.pkl ditemukan!")
else:
    st.error("File model.pkl TIDAK ADA di repository!")
