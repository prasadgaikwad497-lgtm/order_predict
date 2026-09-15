import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
from streamlit_confetti import confetti

# Set page layout and title
st.set_page_config(
    page_title="Customer Predictor",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling and custom button click effect
st.markdown("""
    
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading `model.pkl`: {e}")
    st.stop()

# Header Section
st.markdown("
