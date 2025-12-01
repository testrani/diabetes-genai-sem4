import streamlit as st
import pandas as pd
import numpy as np
import pickle
# Import LogisticRegression explicitly for type hints, though it's not strictly
# needed if the model file already exists.
from sklearn.linear_model import LogisticRegression 

# --- Configuration ---
MODEL_FILE = 'finalized_model.sav'

# Use st.cache_resource to load the model only once.
@st.cache_resource
def load_model(file_path):
    """
    Loads the pre-trained Logistic Regression model from disk.
    Includes error handling for file not found and version mismatch.
    """
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file '{file_path}' not found.")
        st.info("Please ensure you have generated this file by running a training script (e.g., 'train_and_save_model.py') in the same directory.")
        st.stop()
    except Exception as e:
        st.error("Error loading the model.")
        st.info(f"This is often due to a **scikit-learn version mismatch** between when the model was saved and now. Please try retraining and saving the model with your current environment. (Details: {e})")
        st.stop()

# --- Model Loading ---
model = load_model(MODEL_FILE)

# --- Streamlit App Layout ---
st.set_page_config(page_title="Pima Indians Diabetes Prediction", layout="centered", initial_sidebar_state="auto")

st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .main .block-container {
        padding-top: 35px;
        padding-bottom: 35px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)


st.title("🩺 Pima Indians Diabetes Prediction")
st.markdown("Use the inputs below to simulate a patient's diagnostic results and predict their diabetes status using a trained Logistic Regression model.")
st.markdown("---")


# Define the input fields and their typical ranges/defaults
features = {
    'preg': {'label': '1. Pregnancies (Number of times pregnant)', 'min': 0, 'max': 17, 'default': 1, 'step': 1},
    'plas': {'label': '2. Glucose (Plasma glucose concentration)', 'min': 0, 'max': 200, 'default': 120, 'step': 1},
    'pres': {'label': '3. Blood Pressure (Diastolic, mm Hg)', 'min': 0, 'max': 122, 'default': 70, 'step': 1},
    'skin': {'label': '4. Skin Thickness (Triceps fold, mm)', 'min': 0, 'max': 99, 'default': 30, 'step': 1},
    'test': {'label': '5. Insulin (2-Hour serum, mu U/ml)', 'min': 0, 'max': 846, 'default': 0, 'step': 1},
    'mass': {'label': '6. BMI (Body mass index)', 'min': 0.0, 'max': 67.1, 'default': 32.5, 'step': 0.1},
    'pedi': {'label': '7. Diabetes Pedigree Function', 'min': 0.0, 'max': 2.42, 'default': 0.5, 'step': 0.001},
    'age': {'label': '8. Age (years)', 'min': 21, 'max': 81, 'default': 30, 'step': 1},
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

if st.button('Predict Diabetes Status', use_container_width=True):
    # Convert inputs to a numpy array for the model
    input_data = np.array([list(user_inputs.values())])
    
    # NOTE: The Logistic Regression model was saved WITHOUT the StandardScaler.
    # If the model had been saved WITH the StandardScaler, the input data would 
    # need to be scaled here. Assuming the saved model expects raw features.
    
    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    
    st.subheader("Prediction Result")
    
    # The output from the model is 0 for no diabetes, 1 for diabetes
    if prediction[0] == 1.0:
        st.error(f"The model predicts: **POSITIVE for Diabetes**.")
        probability = prediction_proba[0][1]
        st.progress(probability)
        st.write(f"Confidence (Probability of Diabetes): **{probability:.2f}**")
    else:
        st.success(f"The model predicts: **NEGATIVE for Diabetes**.")
        probability = prediction_proba[0][0]
        st.progress(probability)
        st.write(f"Confidence (Probability of No Diabetes): **{probability:.2f}**")

# Optional: Show the input data used for prediction
with st.expander("View Input Data"):
    input_df = pd.DataFrame(user_inputs, index=['Input Values']).T
    input_df.columns = ['Feature Value']
    st.dataframe(input_df)

st.markdown("---")
st.caption(f"Model loaded: {MODEL_FILE}. Prediction made using **{type(model).__name__}**.")
