import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="RandomForest Prediction App", page_icon="🌲", layout="centered"
)


# Model loading logic with caching
@st.cache_resource
def load_model(file_path: str):
    if not os.path.exists(file_path):
        st.error(
            f"Model file '{file_path}' not found in the application directory."
        )
        return None
    try:
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model file: {e}")
        return None


def main():
    st.title("🌲 RandomForest Model Deployment")
    st.write(
        "Enter the required input features below to make predictions using the trained scikit-learn model."
    )

    model = load_model("random.pkl")
    if model is None:
        st.stop()

    # Form to input features matching expected order:
    # ['Gender', 'Marital Status', 'Occupation', 'Monthly Income', 'Educational Qualifications', 'Family size', 'Customer Type']
    st.subheader("Input Parameters")

    with st.form("prediction_form"):
        # Categorical features encoded numerically as required by scikit-learn models
        gender = st.selectbox(
            "Gender",
            options=[0, 1],
            format_func=lambda x: "Male" if x == 1 else "Female",
        )
        marital_status = st.selectbox(
            "Marital Status",
            options=[0, 1, 2],
            format_func=lambda x: {0: "Single", 1: "Married", 2: "Other"}[x],
        )
        occupation = st.selectbox(
            "Occupation",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "Student",
                1: "Employed",
                2: "Self-Employed",
                3: "Unemployed",
            }[x],
        )
        monthly_income = st.number_input(
            "Monthly Income", min_value=0.0, value=50000.0, step=1000.0
        )
        educational_qualifications = st.selectbox(
            "Educational Qualifications",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "Undergraduate",
                1: "Postgraduate",
                2: "Doctorate",
                3: "School",
            }[x],
        )
        family_size = st.slider("Family Size", min_value=1, max_value=15, value=4)
        customer_type = st.selectbox(
            "Customer Type",
            options=[0, 1],
            format_func=lambda x: "Existing" if x == 1 else "New",
        )

        submitted = st.form_submit_button("Predict")

    if submitted:
        # Build pandas DataFrame with exact feature names and sequence
        input_data = pd.DataFrame(
            [
                [
                    gender,
                    marital_status,
                    occupation,
                    monthly_income,
                    educational_qualifications,
                    family_size,
                    customer_type,
                ]
            ],
            columns=[
                "Gender",
                "Marital Status",
                "Occupation",
                "Monthly Income",
                "Educational Qualifications",
                "Family size",
                "Customer Type",
            ],
        )

        try:
            prediction = model.predict(input_data)

            # Optional prediction probabilities
            prob = None
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_data)

            st.success(f"**Prediction Result:** {prediction[0]}")

            if prob is not None:
                st.write("**Prediction Probabilities:**")
                st.write(prob)

        except Exception as e:
            st.error(f"Error during prediction execution: {e}")


if __name__ == "__main__":
    main()
