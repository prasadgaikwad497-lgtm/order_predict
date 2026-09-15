import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time

# Page configuration
st.set_page_config(
    page_title="Customer Classification App",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS for custom styling
st.markdown("""
    
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model.pkl: {e}")
    st.stop()

# Header layout
st.title("🤖 Customer Prediction Portal")
st.markdown("Enter customer metrics below to predict their classification category.")
st.divider()

# Input Form
with st.form("prediction_form"):
    st.subheader("📋 Customer Attributes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        marital_status = st.selectbox("Marital Status", options=[0, 1, 2], format_func=lambda x: ["Single", "Married", "Other"][x])
        occupation = st.selectbox("Occupation", options=[0, 1, 2, 3, 4], format_func=lambda x: f"Occupation Type {x}")
        monthly_income = st.number_input("Monthly Income", min_value=0, value=25000, step=1000)

    with col2:
        edu_qual = st.selectbox("Educational Qualifications", options=[0, 1, 2, 3], format_func=lambda x: f"Level {x}")
        family_size = st.number_input("Family Size", min_value=1, max_value=20, value=3, step=1)
        customer_type = st.selectbox("Customer Type", options=[0, 1, 2], format_func=lambda x: f"Type {x}")

    st.markdown("---")
    submit_button = st.form_submit_button(label="🎯 Predict Customer Class")

# Prediction handling and animation effects
if submit_button:
    # Prepare input array
    input_data = pd.DataFrame([[
        gender, marital_status, occupation, monthly_income, 
        edu_qual, family_size, customer_type
    ]], columns=[
        "Gender", "Marital Status", "Occupation", "Monthly Income", 
        "Educational Qualifications", "Family size", "Customer Type"
    ])

    # Loading animation effect
    with st.spinner("Processing prediction model..."):
        time.sleep(1)  # Visual delay effect
        prediction = model.predict(input_data)[0]
        
        # Probabilities if supported by the model
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
        else:
            probabilities = None

    # Confetti celebration effect
    st.balloons()

    # Display prediction result
    st.success("✅ Prediction Complete!")
    
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric(label="Predicted Class", value=f"Class {prediction}")
    
    if probabilities is not None:
        with res_col2:
            st.metric(label="Prediction Confidence", value=f"{np.max(probabilities) * 100:.1f}%")
