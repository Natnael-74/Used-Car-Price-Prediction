import streamlit as st




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