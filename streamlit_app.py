# app.py

import streamlit as st
import joblib
import pandas as pd  # Import pandas to fix the feature names issue

# Load the trained model
model = joblib.load('car_price_model.pkl')

# Set up the title and description
st.title("Car Price Prediction App")
st.write("Enter the car details to predict its price.")

# Create input fields for user input
age = st.number_input("Car Age (years)", min_value=0, max_value=50, value=0)
mileage = st.number_input("Car Mileage (km)", min_value=0, max_value=200000, value=0)

# Predict button
if st.button("Predict Price"):
    # Create a DataFrame to match the feature names used in training
    input_data = pd.DataFrame([[age, mileage]], columns=['age', 'mileage'])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display the result
    st.success(f"The predicted price of the car is ${prediction:.2f}")
    