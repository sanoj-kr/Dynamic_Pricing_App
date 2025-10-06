# ui_app.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

# Load trained model & transformer (make sure you save them first after training)
model = joblib.load("model.pkl")
transformer = joblib.load("transformer.pkl")

st.title("🚖 Dynamic Pricing Prediction")
st.write("Enter ride details to get the predicted price")

# Input form
with st.form("fare_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        riders = st.number_input("Number of Riders", min_value=0, value=50)
        drivers = st.number_input("Number of Drivers", min_value=0, value=50)
        duration = st.slider("Expected Ride Duration (mins)", 1, 120, 30)
        past_rides = st.number_input("Number of Past Rides", min_value=0, value=10)
        ratings = st.slider("Average Ratings", 1.0, 5.0, 4.5)

    with col2:
        location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
        loyalty = st.selectbox("Customer Loyalty Status", ['Regular', 'Silver', 'Gold'])
        time = st.selectbox("Time of Booking", ["Morning", "Afternoon", "Evening", "Night"])
        vehicle = st.selectbox("Vehicle Type", ["Economy", "Premium"])
    
    submit = st.form_submit_button("Predict Fare")

# Predict button
if submit:
    data = pd.DataFrame([{
        "Number_of_Riders": riders,
        "Number_of_Drivers": drivers,
        "Location_Category": location,
        "Average_Ratings": ratings,
        "Customer_Loyalty_Status": loyalty,
        "Number_of_Past_Rides": past_rides,
        "Time_of_Booking": time,
        "Vehicle_Type": vehicle,
        "Expected_Ride_Duration": duration
    }])
    
    X = transformer.transform(data)
    prediction = model.predict(X)[0]
    st.success(f"Predicted Ride Price: ₹{prediction:.2f}")
