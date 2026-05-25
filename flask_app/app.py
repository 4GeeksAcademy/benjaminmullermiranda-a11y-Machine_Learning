from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "models", "random_forest_diabetes.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler_diabetes.pkl"))

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

FEATURE_LABELS = {
    "Pregnancies": "Número de embarazos",
    "Glucose": "Glucosa (mg/dL)",
    "BloodPressure": "Presión arterial (mm Hg)",
    "SkinThickness": "Grosor de piel (mm)",
    "Insulin": "Insulina (μU/mL)",
    "BMI": "Índice de masa corporal",
    "DiabetesPedigreeFunction": "Función pedigrí de diabetes",
    "Age": "Edad (años)",
}

FEATURE_DEFAULTS = {
    "Pregnancies": 3,
    "Glucose": 120,
    "BloodPressure": 72,
    "SkinThickness": 29,
    "Insulin": 80,
    "BMI": 32.0,
    "DiabetesPedigreeFunction": 0.47,
    "Age": 33,
}

FEATURE_STEPS = {
    "Pregnancies": 1,
    "Glucose": 1,
    "BloodPressure": 1,
    "SkinThickness": 1,
    "Insulin": 1,
    "BMI": 0.1,
    "DiabetesPedigreeFunction": 0.01,
    "Age": 1,
}


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        features=FEATURES,
        labels=FEATURE_LABELS,
        defaults=FEATURE_DEFAULTS,
        steps=FEATURE_STEPS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = [float(request.form[f]) for f in FEATURES]
    except (KeyError, ValueError):
        return render_template("index.html", error="Por favor, completa todos los campos correctamente.",
                               features=FEATURES, labels=FEATURE_LABELS,
                               defaults=FEATURE_DEFAULTS, steps=FEATURE_STEPS)

    X = pd.DataFrame([values], columns=FEATURES)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=FEATURES)
    prediction = int(model.predict(X_scaled)[0])
    probability = float(model.predict_proba(X_scaled)[0][1])

    user_input = dict(zip(FEATURES, values))

    return render_template(
        "result.html",
        prediction=prediction,
        probability=round(probability * 100, 1),
        user_input=user_input,
        labels=FEATURE_LABELS,
    )


if __name__ == "__main__":
    app.run(debug=True)
