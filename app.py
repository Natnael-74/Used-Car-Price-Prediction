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


# 5. Prediction

if st.button("Calculate Market Value"):
    input_dict = {
        "mileage_kmpl": mileage,
        "engine_cc": engine,
        "owner_count": owners,
        "accidents_reported": accidents,
        "car_age": age
    }

    df_input = pd.DataFrame([input_dict])
    df_input[f"brand_{brand}"] = 1
    df_input[f"fuel_type_{fuel}"] = 1
    df_input[f"transmission_{transmission}"] = 1
    df_input[f"service_history_{service}"] = 1

    df_final = df_input.reindex(columns=features, fill_value=0)
    # Ensure features list is a plain Python list (robustness for saved types)
    if not isinstance(features, list):
        try:
            features = list(features)
        except Exception:
            features = [str(f) for f in features]

    # Reindex again with the cleaned features and ensure numeric dtype
    df_final = df_input.reindex(columns=features, fill_value=0)
    df_final = df_final.astype(float)

    # Debug: show aligned input and scaler internals to help diagnose mismatches
    st.write("__Debug: Aligned input (unscaled)__")
    st.write(df_final.T)
    if hasattr(scaler, "mean_"):
        st.write("__Debug: scaler.mean_ (first 10)__", list(scaler.mean_[:10]))
    if hasattr(scaler, "scale_"):
        st.write("__Debug: scaler.scale_ (first 10)__", list(scaler.scale_[:10]))

    # Now scale and predict
    X_scaled = scaler.transform(df_final)

    # Predict
    prediction = model.predict(X_scaled)[0]

    # Result
    st.markdown("---")
    st.markdown(f"""
        <div class="prediction-box">
            <h2 style='color: #28a745;'>Estimated Value</h2>
            <h1 style='font-size: 50px;'>${max(0, prediction):,.2f}</h1>
            <p>Based on current market trends and vehicle condition.</p>
        </div>
    """, unsafe_allow_html=True)
    
    if prediction < 0:
        st.warning("Note: The model predicts a negligible value based on the age/condition provided.")