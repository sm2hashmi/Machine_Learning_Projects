# ============================================
# app_rf.py - Random Forest on Heart Disease
# 13 features from the TensorFlow heart dataset.
# Shows feature importance bar chart.
# ============================================

import streamlit as st
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Random Forest - Heart Disease", layout="wide")
st.title("🌲 Random Forest Model")
st.markdown("Predicts heart disease risk using an ensemble of 100 decision trees.")
st.markdown("**Dataset:** Cleveland Heart Disease (TensorFlow) – 13 features")

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'model_rf.joblib')
    model = joblib.load(model_path)
    return model

model = load_model()
st.success("✅ Model loaded successfully!")

# ---------- DEFINE FEATURES ----------
# 13 features in the exact order the model expects
feature_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

# Display names and UI types
features_ui = [
    ("Age (years)", "slider", 20, 80, 50),
    ("Sex", "select", ["Female", "Male"]),
    ("Chest Pain Type", "select", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]),
    ("Resting BP (mmHg)", "slider", 80, 200, 120),
    ("Cholesterol (mg/dl)", "slider", 100, 400, 200),
    ("Fasting BS > 120 mg/dl", "select", ["False", "True"]),
    ("Resting ECG", "select", ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"]),
    ("Max Heart Rate", "slider", 70, 220, 150),
    ("Exercise Angina", "select", ["No", "Yes"]),
    ("ST Depression", "slider", 0.0, 6.0, 1.0),
    ("Slope of ST", "select", ["Upsloping", "Flat", "Downsloping"]),
    ("Major Vessels (ca)", "slider", 0, 4, 0),
    ("Thalassemia", "select", ["Normal", "Fixed Defect", "Reversible Defect", "Not Described"]),
]

# ---------- SIDEBAR INPUTS ----------
st.sidebar.header("Patient Data Input")
st.sidebar.markdown("Adjust the values below:")

input_values = []
for i, ui in enumerate(features_ui):
    display_name = ui[0]
    ui_type = ui[1]
    if ui_type == "slider":
        min_val = ui[2]
        max_val = ui[3]
        default = ui[4]
        value = st.sidebar.slider(display_name, min_val, max_val, default, step=1.0 if isinstance(default, float) else 1)
    elif ui_type == "select":
        options = ui[2]
        selected = st.sidebar.selectbox(display_name, options)
        value = options.index(selected)  # store index as integer
    input_values.append(value)

# Convert to numpy array (1 sample, 13 features)
input_array = np.array(input_values).reshape(1, -1)

# ---------- PREDICTION ----------
if st.button("🔮 Predict", type="primary"):
    # Predict
    prediction = model.predict(input_array)[0]
    probabilities = model.predict_proba(input_array)[0]
    confidence = probabilities[prediction]
    
    # Display result
    col1, col2 = st.columns(2)
    with col1:
        if prediction == 1:
            st.error("### ❌ Heart Disease Detected")
            st.write("The model predicts a **high risk** of heart disease.")
        else:
            st.success("### ✅ No Heart Disease Detected")
            st.write("The model predicts a **low risk** of heart disease.")
    with col2:
        st.metric("Confidence", f"{confidence * 100:.1f}%")
    
    # ---------- FEATURE IMPORTANCE ----------
    st.subheader("📊 Feature Importance")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(sorted_names, sorted_importances, color='skyblue')
    ax.set_xlabel("Importance Score")
    ax.set_title("Random Forest Feature Importance")
    ax.invert_yaxis()
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', va='center', fontsize=9)
    st.pyplot(fig)
    
    # ---------- INPUT SUMMARY ----------
    with st.expander("📋 View Patient Input Values"):
        for name, value in zip(feature_names, input_values):
            st.write(f"**{name}**: {value}")