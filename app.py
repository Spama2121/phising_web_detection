import streamlit as st
import joblib
import pandas as pd
from scipy.io import arff
import os

 
# CONFIG
 
st.set_page_config(
    page_title="Phishing Website Detection",
    page_icon=" ",
    layout="centered"
)

st.title("  Phishing Website Detection")
st.write("Simulasi deteksi phishing berbasis dataset ")

MODEL_PATH = "phishing_random_forest_model.pkl"
DATASET_PATH = "Training Dataset.arff"

 
# LOAD MODEL
 
if not os.path.exists(MODEL_PATH):
    st.error("  Model tidak ditemukan")
    st.stop()

model = joblib.load(MODEL_PATH)

 
# LOAD DATASET
 
if not os.path.exists(DATASET_PATH):
    st.error("  Dataset tidak ditemukan")
    st.stop()

data, meta = arff.loadarff(DATASET_PATH)
df = pd.DataFrame(data)

# Decode byte columns
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].apply(lambda x: x.decode("utf-8"))

 
# PREPARE DATA
 
X = df.drop("Result", axis=1).astype(int)
y_true = df["Result"].astype(int)
y_pred = model.predict(X)

df["Prediction"] = y_pred
df["Correct"] = df["Result"] == df["Prediction"]

 
# DATASET INFO
 
st.subheader(" Informasi Dataset")
st.write(f"Jumlah data: **{df.shape[0]}**")
st.write(f"Jumlah fitur: **{df.shape[1] - 2}**")

 
# MODE SELECTION
 
mode = st.radio(
    "Pilih mode simulasi:",
    ["Simulasi Data Acak", "Simulasi Data Salah Prediksi"]
)

 
# MODE 1: RANDOM DATA
 
if mode == "Simulasi Data Acak":
    index = st.slider(
        "Pilih index data",
        min_value=0,
        max_value=len(df) - 1,
        value=0
    )

    sample = df.iloc[index]
    X_sample = sample.drop(["Result", "Prediction", "Correct"]).values.reshape(1, -1)

    pred = model.predict(X_sample)[0]

    st.markdown("Hasil Prediksi")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Label Asli")
        st.success("LEGITIMATE" if sample["Result"] == 1 else "PHISHING")

    with col2:
        st.write("Prediksi Model")
        st.success("LEGITIMATE" if pred == 1 else "PHISHING")


    st.dataframe(sample)

 
# MODE 2: MISCLASSIFIED DATA
 
else:
    ERROR_INDEXES = [
        22, 111, 128, 187, 195, 236, 249, 254, 285, 355,
        384, 427, 442, 446, 448, 479, 588, 684, 699, 716
    ]

    st.write(f"Jumlah data salah prediksi: **{len(ERROR_INDEXES)}**")

    selected_index = st.selectbox(
        "Pilih index data salah prediksi",
        ERROR_INDEXES
    )

    sample = df.loc[selected_index]
    X_sample = sample.drop(["Result", "Prediction", "Correct"]).values.reshape(1, -1)

    st.markdown("Kesalahan Prediksi")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Label Asli")
        st.error("PHISHING" if sample["Result"] == -1 else "LEGITIMATE")

    with col2:
        st.write("Prediksi Model")
        st.error("PHISHING" if sample["Prediction"] == -1 else "LEGITIMATE")

    st.dataframe(sample)

 
# FOOTER
 
st.markdown("---")
st.caption("UAS Big Data & Data Mining | Aldo_25.21.15939 | AMIKOM Yogyakarta")
