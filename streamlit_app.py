import streamlit as st
import joblib
import numpy as np


rf_model = joblib.load("rf_model.pkl")
dt_model = joblib.load("dt_model.pkl")
knn_model = joblib.load("knn_model.pkl")
nb_model = joblib.load("nb_model.pkl")

st.title("Obesity Level Prediction")

age = st.number_input("Age")
weight = st.number_input("Weight")
height = st.number_input("Height")

input_data = np.array([[age, weight, height]])

model_choice = st.selectbox(
    "Choose Model",
    ["Random Forest", "Decision Tree", "KNN", "Logistic Regression"]
)

if model_choice == "Random Forest":
    model = rf_model
elif model_choice == "Decision Tree":
    model = dt_model
elif model_choice == "KNN":
    model = knn_model
else:
    model = lr_model

if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Obesity Level: {prediction[0]}")
