# ============================================
# app_dt.py - Decision Tree on Pima Indians Diabetes
# 8 features from UCI diabetes dataset.
# Shows the actual Decision Tree plot.
# ============================================

import streamlit as st
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Decision Tree - Diabetes", layout="wide")
st.title("🌳 Decision Tree Model")
st.markdown("Predicts diabetes onset based on patient health metrics.")
st.markdown("**Dataset:** Pima Indians Diabetes (UCI) – 8 features")

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'model_dt.joblib')
    model = joblib.load(model_path)
    return model

model = load_model()
st.success("✅ Model loaded successfully!")

# ---------- DEFINE FEATURES (8) ----------
feature_names = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

# UI definitions: (display_name, type, min, max, default)
features_ui = [
    ("Number of Pregnancies", "slider", 0, 17, 3),
    ("Glucose (mg/dL)", "slider", 0, 200, 120),
    ("Blood Pressure (mm Hg)", "slider", 0, 130, 70),
    ("Skin Thickness (mm)", "slider", 0, 100, 25),
    ("Insulin (μIU/mL)", "slider", 0, 850, 100),
    ("BMI (kg/m²)", "slider", 0.0, 60.0, 30.0),
    ("Diabetes Pedigree Function", "slider", 0.0, 2.5, 0.5),
    ("Age (years)", "slider", 20, 100, 30),
]

# ---------- SIDEBAR INPUTS ----------
st.sidebar.header("Patient Data Input")
st.sidebar.markdown("Adjust the health metrics below:")

input_values = []
for ui in features_ui:
    display_name = ui[0]
    ui_type = ui[1]
    if ui_type == "slider":
        min_val = ui[2]
        max_val = ui[3]
        default = ui[4]
        # Handle integer vs float
        if isinstance(default, float):
            value = st.sidebar.slider(display_name, float(min_val), float(max_val), float(default), step=0.1)
        else:
            value = st.sidebar.slider(display_name, int(min_val), int(max_val), int(default), step=1)
    input_values.append(value)

input_array = np.array(input_values).reshape(1, -1)

# ---------- PREDICTION ----------
if st.button("🔮 Predict", type="primary"):
    # Predict (no scaling needed)
    prediction = model.predict(input_array)[0]
    probabilities = model.predict_proba(input_array)[0]
    confidence = probabilities[prediction]
    
    # Display result
    col1, col2 = st.columns(2)
    with col1:
        if prediction == 1:
            st.error("### ❌ Diabetes Detected")
            st.write("The model predicts the patient has **diabetes**.")
        else:
            st.success("### ✅ No Diabetes Detected")
            st.write("The model predicts the patient does **not** have diabetes.")
    with col2:
        st.metric("Confidence", f"{confidence * 100:.1f}%")
    
    # ---------- VISUALIZE DECISION TREE ----------
    st.subheader("🌿 The Decision Tree")
    st.markdown("""
    Follow the path from the top (root) down to a leaf. 
    Each node asks a question about a feature. 
    The tree is **max depth = 4**, making it easy to interpret.
    """)
    
    fig, ax = plt.subplots(figsize=(16, 10))
    plot_tree(
        model,
        filled=True,
        feature_names=feature_names,
        class_names=["No Diabetes", "Diabetes"],
        rounded=True,
        proportion=True,
        max_depth=4,
        fontsize=10,
        ax=ax
    )
    st.pyplot(fig)
    
    # ---------- EXPLAIN THE DECISION PATH ----------
    st.subheader("🧭 Path Taken for Your Prediction")
    st.markdown("These are the decisions the tree made based on your inputs:")
    
    decision_path = model.decision_path(input_array)
    node_indices = decision_path.indices
    
    for i, node_id in enumerate(node_indices):
        # If it's not a leaf node, show the split condition
        if model.tree_.feature[node_id] != -2:
            feature_name = feature_names[model.tree_.feature[node_id]]
            threshold = model.tree_.threshold[node_id]
            value = input_array[0][model.tree_.feature[node_id]]
            go_left = value <= threshold
            direction = "Yes" if go_left else "No"
            st.write(f"**Step {i+1}**: Is **{feature_name}** ≤ {threshold:.2f}? → **{direction}** (your value: {value:.2f})")
        else:
            st.write(f"**Step {i+1}**: Reached a leaf node → final decision is **{'Diabetes' if prediction==1 else 'No Diabetes'}**")
    
    # ---------- INPUT SUMMARY ----------
    with st.expander("📋 View Patient Input Values"):
        for name, value in zip(feature_names, input_values):
            st.write(f"**{name}**: {value}")