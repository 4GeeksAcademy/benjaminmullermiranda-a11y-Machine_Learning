import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "models", "random_forest_diabetes.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler_diabetes.pkl"))

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

st.set_page_config(
    page_title="Predictor de Diabetes",
    page_icon="🩺",
    layout="centered",
)

st.title("🩺 Predictor de Diabetes")
st.markdown(
    "Introduce tus datos clínicos y el modelo de **Random Forest** estimará tu riesgo de diabetes."
)
st.divider()

st.subheader("Datos del paciente")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Número de embarazos", min_value=0, max_value=20, value=3, step=1)
    glucose = st.number_input("Glucosa (mg/dL)", min_value=0, max_value=300, value=120, step=1)
    blood_pressure = st.number_input("Presión arterial (mm Hg)", min_value=0, max_value=200, value=72, step=1)
    skin_thickness = st.number_input("Grosor de piel (mm)", min_value=0, max_value=100, value=29, step=1)

with col2:
    insulin = st.number_input("Insulina (μU/mL)", min_value=0, max_value=900, value=80, step=1)
    bmi = st.number_input("Índice de masa corporal (BMI)", min_value=0.0, max_value=70.0, value=32.0, step=0.1, format="%.1f")
    dpf = st.number_input("Función pedigrí de diabetes", min_value=0.0, max_value=3.0, value=0.47, step=0.01, format="%.2f")
    age = st.number_input("Edad (años)", min_value=1, max_value=120, value=33, step=1)

st.divider()

if st.button("Analizar riesgo →", use_container_width=True, type="primary"):
    values = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
    X = pd.DataFrame([values], columns=FEATURES)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=FEATURES)

    prediction = int(model.predict(X_scaled)[0])
    probability = float(model.predict_proba(X_scaled)[0][1])

    st.divider()

    if prediction == 1:
        st.error(f"⚠️ **Alto riesgo de diabetes** — probabilidad estimada: **{probability*100:.1f}%**")
    else:
        st.success(f"✅ **Bajo riesgo de diabetes** — probabilidad estimada: **{probability*100:.1f}%**")

    st.progress(probability, text=f"Probabilidad: {probability*100:.1f}%")

    with st.expander("Ver datos introducidos"):
        labels = {
            "Pregnancies": "Embarazos",
            "Glucose": "Glucosa (mg/dL)",
            "BloodPressure": "Presión arterial",
            "SkinThickness": "Grosor de piel",
            "Insulin": "Insulina",
            "BMI": "BMI",
            "DiabetesPedigreeFunction": "Función pedigrí",
            "Age": "Edad",
        }
        resumen = {labels[f]: v for f, v in zip(FEATURES, values)}
        st.table(pd.DataFrame(resumen.items(), columns=["Variable", "Valor"]))

st.markdown(
    "<p style='text-align:center; color:gray; font-size:0.8rem;'>"
    "Modelo Random Forest entrenado con el dataset Pima Indians Diabetes · "
    "Solo orientativo, no sustituye el diagnóstico médico.</p>",
    unsafe_allow_html=True,
)
