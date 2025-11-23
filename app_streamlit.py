import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load("xgb_model.pkl")

# App title
st.title("Air Quality Health Impact Prediction")
st.write("Enter feature values to predict Health Impact Class.")

# --------------------
# User input (replace with your actual feature names)
# --------------------
feature1 = st.number_input("Feature1")
feature2 = st.number_input("Feature2")
feature3 = st.number_input("Feature3")
feature4 = st.number_input("Feature4")
feature5 = st.number_input("Feature5")
# Add more features as needed, in the same order as the model

# Create DataFrame for prediction
input_data = pd.DataFrame({
    'Feature1': [feature1],
    'Feature2': [feature2],
    'Feature3': [feature3],
    'Feature4': [feature4],
    'Feature5': [feature5],
    # Add more features if necessary
})

# Prediction button
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)
        st.success(f"Predicted Health Impact Class: {prediction[0]}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")

# Optional: Show input data
st.subheader("Input Data")
st.write(input_data)
