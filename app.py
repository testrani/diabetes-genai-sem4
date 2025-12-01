import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

# --- Configuration ---
# NOTE: This app assumes you have the 'finalized_model.sav' file
# and the 'pima-indians-diabetes.csv' file in the same directory.
MODEL_FILE = 'finalized_model.sav'

def load_model(file_path):
    """Loads the pre-trained model from disk."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file '{file_path}' not found.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Load the model once when the app starts
model = load_model(MODEL_FILE)

# --- Streamlit App Layout ---
st.set_page_config(page_title="Pima Indians Diabetes Prediction", layout="centered")

st.title("🩺 Diabetes Prediction App")
st.markdown("Enter the patient's data below to get a diabetes prediction.")
st.markdown("---")


# Define the input fields and their typical ranges/defaults
features = {
    'preg': {'label': 'Pregnancies (Number of times pregnant)', 'min': 0, 'max': 17, 'default': 1, 'step': 1},
    'plas': {'label': 'Glucose (Plasma glucose concentration a 2 hours in an oral glucose tolerance test)', 'min': 0, 'max': 200, 'default': 120, 'step': 1},
    'pres': {'label': 'Blood Pressure (Diastolic blood pressure (mm Hg))', 'min': 0, 'max': 122, 'default': 70, 'step': 1},
    'skin': {'label': 'Skin Thickness (Triceps skin fold thickness (mm))', 'min': 0, 'max': 99, 'default': 30, 'step': 1},
    'test': {'label': 'Insulin (2-Hour serum insulin (mu U/ml))', 'min': 0, 'max': 846, 'default': 0, 'step': 1},
    'mass': {'label': 'BMI (Body mass index (weight in kg/(height in m)^2))', 'min': 0.0, 'max': 67.1, 'default': 32.5, 'step': 0.1},
    'pedi': {'label': 'Diabetes Pedigree Function', 'min': 0.0, 'max': 2.42, 'default': 0.5, 'step': 0.001},
    'age': {'label': 'Age (years)', 'min': 21, 'max': 81, 'default': 30, 'step': 1},
}

# Use two columns for cleaner input layout
col1, col2 = st.columns(2)
user_inputs = {}

# Create input widgets
input_keys = list(features.keys())
for i, key in enumerate(input_keys):
    feature = features[key]
    
    # Place inputs in alternating columns
    target_col = col1 if i % 2 == 0 else col2
    
    if key in ['mass', 'pedi']: # Use number_input for floats
        user_inputs[key] = target_col.number_input(
            label=feature['label'],
            min_value=feature['min'],
            max_value=feature['max'],
            value=feature['default'],
            step=feature['step'],
            format="%.3f" if key == 'pedi' else "%.1f"
        )
    else: # Use number_input for integers
        user_inputs[key] = target_col.number_input(
            label=feature['label'],
            min_value=feature['min'],
            max_value=feature['max'],
            value=feature['default'],
            step=feature['step']
        )

# --- Prediction Logic ---
st.markdown("---")

if st.button('Predict Diabetes Status'):
    # Convert inputs to a numpy array for the model
    input_data = np.array([list(user_inputs.values())])
    
    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    
    st.subheader("Prediction Result")
    
    if prediction[0] == 1.0:
        st.error(f"The patient is **predicted to have Diabetes**.")
        st.progress(prediction_proba[0][1])
        st.write(f"Confidence (Probability of Diabetes): **{prediction_proba[0][1]:.2f}**")
    else:
        st.success(f"The patient is **predicted NOT to have Diabetes**.")
        st.progress(prediction_proba[0][0])
        st.write(f"Confidence (Probability of No Diabetes): **{prediction_proba[0][0]:.2f}**")

# Optional: Show the input data used for prediction
with st.expander("View Input Data"):
    input_df = pd.DataFrame(user_inputs, index=['Input Values']).T
    input_df.columns = ['Feature Value']
    st.dataframe(input_df)
