# app.py
import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("model.pkl")

st.title("🌸 Iris Flower Prediction App")

# Input fields
sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.1)
sepal_width = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5)
petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4)
petal_width = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2)

if st.button("Predict"):
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(data)[0]
    iris_classes = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"🌼 Predicted Iris Flower: **{iris_classes[prediction]}**")
