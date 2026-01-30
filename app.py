import streamlit as st
import pandas as pd
import joblib
import numpy as np



# title
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS 
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    features = joblib.load("models/features.pkl")
    return model, scaler, features

try:
    model, scaler, features = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()


#  Sidebar

st.sidebar.title(" 🚘 About CarValue")
st.sidebar.info(
    "This AI model analyzes mileage, engine capacity, and history "
    "to provide a fair market valuation for used vehicles."
)


#  Input Form

st.title("🚗 Used Car Price Predictor")
st.markdown("Enter the vehicle details below to get an instant valuation.")

with st.container():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📋 Specifications")
        brand = st.selectbox("Brand", ["Toyota", "Honda", "Hyundai", "Ford", "Chevrolet", "Nissan", "Volkswagen", "Kia", "BMW", "Tesla"])
        fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric"])
        transmission = st.selectbox("Transmission", ["Automatic", "Manual"])

    with col2:
        st.subheader("⚙️ Technical")
        mileage = st.number_input("Mileage (km/l)", 5.0, 50.0, 18.0)
        engine = st.number_input("Engine (cc)", 600, 6000, 1500)
        owners = st.slider("Previous Owners", 1, 5, 1)

    with col3:
        st.subheader("📜 History")
        age = st.number_input("Car Age (Years)", 0, 30, 5)
        service = st.selectbox("Service History", ["Full", "Partial", "Unknown"])
        accidents = st.radio("Accidents Reported?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
