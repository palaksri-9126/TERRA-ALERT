import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# -----------------------------
# Load trained models & encoder
# -----------------------------
rf_model = joblib.load("models/random_forest_model.pkl")
xgb_model = joblib.load("models/xgboost_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="🌋 TERRA - ALERT",
    page_icon="🌧️",
    layout="wide",
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg, #111827, #1f2937);
            color: white;
        }
        h1, h2, h3, h4, h5 {
            color: #f9fafb;
        }
        .stButton>button {
            background: linear-gradient(90deg, #10b981, #059669);
            color: white;
            border-radius: 10px;
            padding: 0.6em 1.5em;
            font-weight: 600;
            border: none;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #059669, #047857);
        }
        .risk-box {
            text-align: center;
            padding: 20px;
            border-radius: 15px;
            color: white;
            font-size: 1.3em;
            font-weight: bold;
            animation: fadeIn 1.5s;
        }
        .High { background-color: #dc2626; }
        .Medium { background-color: #f59e0b; }
        .Low { background-color: #16a34a; }
        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("🧩 Input Parameters")
st.sidebar.markdown("Adjust the environmental factors below to estimate the likelihood of a landslide.")

rainfall = st.sidebar.number_input("🌧️ Rainfall (mm)", min_value=0.0, step=0.1)
slope = st.sidebar.number_input("⛰️ Slope Angle (°)", min_value=0.0, max_value=90.0, step=0.1)
soil = st.sidebar.number_input("💧 Soil Saturation (%)", min_value=0.0, max_value=100.0, step=0.1)
vegetation = st.sidebar.number_input("🌿 Vegetation Cover (%)", min_value=0.0, max_value=100.0, step=0.1)

# -----------------------------
# Header
# -----------------------------
st.title("🌋 TERRA - ALERT")
st.markdown("""
Turn terrain data into actionable insights.  
Stay ahead with real-time, AI-powered predictions.
""")
st.divider()

# -----------------------------
# Predict Button
# -----------------------------
predict = st.button("⚡ Predict Landslide Risk", help="Click to get real-time risk predictions")

if predict:
    # Derived features
    rainfall_slope = rainfall * slope
    rainfall_soil_ratio = rainfall / (soil + 1e-6)
    slope_soil = slope * soil

    # Input DataFrame
    features = pd.DataFrame([[rainfall, slope, soil, vegetation,
                              rainfall_slope, rainfall_soil_ratio, slope_soil]],
                            columns=['Rainfall_mm', 'Slope_Angle', 'Soil_Saturation', 'Vegetation_Cover',
                                     'Rainfall_Slope', 'Rainfall_SoilRatio', 'Slope_Soil'])

    # -----------------------------
    # Real-Time Prediction
    # -----------------------------
    rf_pred = rf_model.predict(features)[0]
    xgb_pred = xgb_model.predict(features)[0]

    rf_label = label_encoder.inverse_transform([rf_pred])[0]
    xgb_label = label_encoder.inverse_transform([xgb_pred])[0]

    # -----------------------------
    # Display Predictions
    # -----------------------------
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<div class='risk-box {rf_label}'>🌲 Random Forest: {rf_label}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='risk-box {xgb_label}'>⚡ XGBoost: {xgb_label}</div>", unsafe_allow_html=True)

    # -----------------------------
    # Confidence Visualization
    # -----------------------------
    rf_probs = rf_model.predict_proba(features)[0]
    xgb_probs = xgb_model.predict_proba(features)[0]

    prob_df = pd.DataFrame({
        "Risk Level": label_encoder.classes_,
        "Random Forest": rf_probs,
        "XGBoost": xgb_probs
    })

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=prob_df["Risk Level"],
        y=prob_df["Random Forest"],
        name="Random Forest",
        marker_color="#34d399"
    ))
    fig.add_trace(go.Bar(
        x=prob_df["Risk Level"],
        y=prob_df["XGBoost"],
        name="XGBoost",
        marker_color="#60a5fa"
    ))
    fig.update_layout(
        title="📊 Model Confidence Levels",
        barmode="group",
        template="plotly_dark",
        yaxis=dict(title="Probability"),
        xaxis=dict(title="Risk Level")
    )
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Input Visualization (Radar Chart)
    # -----------------------------
    st.subheader("📈 Environmental Factor Overview")

    input_df = pd.DataFrame(dict(
        r=[rainfall, slope, soil, vegetation],
        theta=['Rainfall (mm)', 'Slope (°)', 'Soil Saturation (%)', 'Vegetation Cover (%)']
    ))

    fig_input = go.Figure()
    fig_input.add_trace(go.Scatterpolar(
        r=input_df['r'],
        theta=input_df['theta'],
        fill='toself',
        name='Input Values',
        line_color='#10b981'
    ))

    fig_input.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0, max(100, rainfall + 10)])
        ),
        showlegend=False,
        template="plotly_dark",
        title="🧭 Terrain Condition Radar"
    )

    st.plotly_chart(fig_input, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
