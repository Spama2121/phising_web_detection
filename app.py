import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Phishing Website Detection",
    layout="centered"
)

st.title(" Phishing Website Detection")

st.write("Masukkan fitur website untuk prediksi phishing")

# ======================
# LOAD MODEL (AMAN)
# ======================
try:
    model = joblib.load("phishing_random_forest_model.pkl")
    st.success("Model berhasil dimuat")
except Exception as e:
    st.error("Model gagal dimuat")
    st.stop()

# ======================
# INPUT FITUR
# ======================
NUM_FEATURES = model.n_features_in_

st.info(f"Model membutuhkan {NUM_FEATURES} fitur")

input_data = []

for i in range(NUM_FEATURES):
    value = st.number_input(
        f"Feature {i+1}",
        min_value=-1,
        max_value=1,
        value=0
    )
    input_data.append(value)

# ======================
# PREDIKSI
# ======================
if st.button("Predict"):
    try:
        prediction = model.predict([input_data])

        if prediction[0] == -1:
            st.error(" Website terdeteksi PHISHING")
        else:
            st.success(" Website LEGITIMATE")

    except Exception as e:
        st.error("Terjadi kesalahan saat prediksi")
        st.text(str(e))
