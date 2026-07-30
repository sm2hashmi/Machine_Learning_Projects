# ============================================
# app_knn.py - KNN on Breast Cancer (POLISHED UI)
# Features: Wider sidebar + Grouped sliders
# ============================================

import streamlit as st
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="KNN - Breast Cancer", layout="wide")
st.title("📏 K-Nearest Neighbors (KNN)")
st.markdown("Predicts whether a breast tumor is malignant or benign based on 30 cell nucleus measurements.")
st.markdown("**Dataset:** Wisconsin Breast Cancer (sklearn) – 30 features")

# ---------- CUSTOM CSS TO WIDEN THE SIDEBAR ----------
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            min-width: 400px;
            width: 450px;
        }
        .stSlider > div > div > div {
            padding-top: 5px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- LOAD MODEL & SCALER ----------
@st.cache_resource
def load_model_and_scaler():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, 'models', 'model_knn.joblib')
    scaler_path = os.path.join(base_dir, 'models', 'scaler.joblib')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

try:
    model, scaler = load_model_and_scaler()
    st.success(f"✅ Model loaded! Expects {model.n_features_in_} features.")
except Exception as e:
    st.error(f"❌ Failed to load models: {e}")
    st.stop()

# ---------- DEFINE FEATURES (30) ----------
feature_names = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error", "smoothness error",
    "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
]

# Ranges and defaults (all converted to float for safety)
ranges = [
    (5.0, 30.0), (10.0, 40.0), (40.0, 200.0), (150.0, 2500.0), (0.0, 0.2),
    (0.0, 0.3), (0.0, 0.4), (0.0, 0.2), (0.0, 0.3), (0.0, 0.1),
    (0.0, 4.0), (0.0, 4.0), (0.0, 30.0), (0.0, 150.0), (0.0, 0.03),
    (0.0, 0.1), (0.0, 0.2), (0.0, 0.05), (0.0, 0.05), (0.0, 0.03),
    (8.0, 50.0), (15.0, 60.0), (50.0, 300.0), (250.0, 4000.0), (0.0, 0.3),
    (0.0, 0.5), (0.0, 0.6), (0.0, 0.3), (0.0, 0.4), (0.0, 0.15)
]

defaults = [
    14.0, 20.0, 90.0, 600.0, 0.1, 0.1, 0.1, 0.05, 0.2, 0.05,
    0.5, 0.5, 5.0, 30.0, 0.01, 0.02, 0.03, 0.01, 0.01, 0.01,
    18.0, 30.0, 120.0, 900.0, 0.15, 0.2, 0.2, 0.1, 0.25, 0.08
]

# ---------- SIDEBAR INPUTS (GROUPED IN EXPANDERS) ----------
st.sidebar.header("Patient Data Input")
st.sidebar.markdown("Adjust the 30 cell nucleus measurements:")

input_values = []

# Group 1: Mean Values (indices 0-9)
with st.sidebar.expander("📊 Mean Values (10 features)", expanded=True):
    for i in range(0, 10):
        min_val, max_val = ranges[i]
        default = defaults[i]
        val = st.slider(
            label=feature_names[i],
            min_value=min_val,
            max_value=max_val,
            value=default,
            step=0.01,
            key=f"slider_mean_{i}"
        )
        input_values.append(val)

# Group 2: Error Values (indices 10-19)
with st.sidebar.expander("📏 Error Values (10 features)", expanded=False):
    for i in range(10, 20):
        min_val, max_val = ranges[i]
        default = defaults[i]
        val = st.slider(
            label=feature_names[i],
            min_value=min_val,
            max_value=max_val,
            value=default,
            step=0.01,
            key=f"slider_error_{i}"
        )
        input_values.append(val)

# Group 3: Worst Values (indices 20-29)
with st.sidebar.expander("🔥 Worst Values (10 features)", expanded=False):
    for i in range(20, 30):
        min_val, max_val = ranges[i]
        default = defaults[i]
        val = st.slider(
            label=feature_names[i],
            min_value=min_val,
            max_value=max_val,
            value=default,
            step=0.01,
            key=f"slider_worst_{i}"
        )
        input_values.append(val)

# Convert to numpy array (1 sample, 30 features)
input_array = np.array(input_values).reshape(1, -1)

# ---------- PREDICTION ----------
if st.button("🔮 Predict", type="primary"):
    # Scale input
    input_scaled = scaler.transform(input_array)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    confidence = probabilities[prediction]
    
    # Display result
    col1, col2 = st.columns(2)
    with col1:
        if prediction == 1:
            st.error("### ❌ Malignant Tumor Detected")
            st.write("The model predicts the tumor is **malignant**.")
        else:
            st.success("### ✅ Benign Tumor Detected")
            st.write("The model predicts the tumor is **benign**.")
    with col2:
        st.metric("Confidence", f"{confidence * 100:.1f}%")
    
    # ---------- NEIGHBOR DISTANCES ----------
    distances, indices = model.kneighbors(input_scaled, n_neighbors=5)
    distances = distances.flatten()
    
    st.subheader("📏 Distance to 5 Nearest Neighbors")
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    neighbor_labels = [f"Neighbor {i+1}" for i in range(5)]
    bars = ax1.bar(neighbor_labels, distances, color='coral')
    ax1.set_ylabel("Euclidean Distance")
    ax1.set_title("How similar are the 5 closest tumors?")
    for bar, dist in zip(bars, distances):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{dist:.3f}', ha='center', va='bottom', fontsize=10)
    st.pyplot(fig1)
    
    # ---------- PCA SCATTER PLOT ----------
    st.subheader("🗺️ PCA Projection of Training Data")
    st.markdown("""
    The training data is projected into 2D using PCA. 
    **Your patient** is shown as a **red star**. The 5 nearest neighbors are circled in **green**.
    """)
    
    X_train = model._fit_X
    if X_train.shape[0] > 500:
        X_train_plot = X_train[:500]
    else:
        X_train_plot = X_train
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train_plot)
    user_pca = pca.transform(input_scaled)
    neighbor_pca = pca.transform(X_train[indices.flatten()])
    
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.3, c='lightgrey', s=30, label='Training Data')
    ax2.scatter(user_pca[0, 0], user_pca[0, 1], c='red', s=200, marker='*', 
                edgecolors='black', linewidth=2, label='Your Patient', zorder=5)
    ax2.scatter(neighbor_pca[:, 0], neighbor_pca[:, 1], c='green', s=120, 
                edgecolors='darkgreen', linewidth=2, label='5 Nearest Neighbors', zorder=4)
    for nbr in neighbor_pca:
        ax2.plot([user_pca[0, 0], nbr[0]], [user_pca[0, 1], nbr[1]], 
                 'k--', alpha=0.3, linewidth=1)
    ax2.set_xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax2.set_ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax2.set_title('KNN: Your Patient vs. 5 Nearest Neighbors in 2D')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    
    # ---------- INPUT SUMMARY ----------
    with st.expander("📋 View All 30 Input Values"):
        for name, value in zip(feature_names, input_values):
            st.write(f"**{name}**: {value}")