import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page configuration
st.set_page_config(
    page_title="Customer Classifier App",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Customer Classification System")
st.write("Provide input for all 8 customer features below to generate a prediction.")

# Load Model
@st.cache_resource
def load_model():
    try:
        with open("random.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Error: 'random.pkl' file not found in the root directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model is not None:
    # Feature Mappings for Categorical Variables (Numeric Encodings)
    gender_map = {"Male": 0, "Female": 1}
    marital_map = {"Single": 0, "Married": 1, "Divorced": 2, "Widowed": 3}
    occupation_map = {"Student": 0, "Employed": 1, "Unemployed": 2, "Self-Employed": 3, "Retired": 4}
    education_map = {"Undergraduate": 0, "Graduate": 1, "Post Graduate": 2, "School/Other": 3}
    customer_type_map = {"Type A": 0, "Type B": 1, "Type C": 2}

    st.subheader("Customer Input Features")
    
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        gender_str = st.selectbox("Gender", list(gender_map.keys()))
        marital_str = st.selectbox("Marital Status", list(marital_map.keys()))
        occupation_str = st.selectbox("Occupation", list(occupation_map.keys()))

    with col2:
        monthly_income = st.number_input("Monthly Income", min_value=0, max_value=1000000, value=25000, step=1000)
        education_str = st.selectbox("Educational Qualifications", list(education_map.keys()))
        family_size = st.number_input("Family size", min_value=1, max_value=20, value=4, step=1)
        customer_type_str = st.selectbox("Customer Type", list(customer_type_map.keys()))

    # Map categorical inputs to numeric values expected by tree algorithms
    gender = gender_map[gender_str]
    marital_status = marital_map[marital_str]
    occupation = occupation_map[occupation_str]
    educational_qualifications = education_map[education_str]
    customer_type = customer_type_map[customer_type_str]

    # Construct input dataframe with exact feature names and order
    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Marital Status": marital_status,
        "Occupation": occupation,
        "Monthly Income": monthly_income,
        "Educational Qualifications": educational_qualifications,
        "Family size": family_size,
        "Customer Type": customer_type
    }])

    st.markdown("---")

    if st.button("Predict Class", type="primary", use_container_width=True):
        try:
            # Perform prediction
            prediction = model.predict(input_data)[0]
            
            st.success(f"**Predicted Class:** `{prediction}`")

            # Check for probability support
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)[0]
                classes = model.classes_
                
                st.subheader("Prediction Probability")
                prob_df = pd.DataFrame({
                    "Class": classes,
                    "Probability": [f"{p * 100:.2f}%" for p in probabilities]
                })
                st.dataframe(prob_df, hide_index=True, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
