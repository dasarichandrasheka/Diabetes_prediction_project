import streamlit as st
import pandas as pd 
import numpy as np
import joblib

model = joblib.load("model.joblib")
scaler= joblib.load("scaler.joblib")

st.title("diabetes prediction  app")

st.write("enter the details below to predict whther  the person has diabetes.")

preg = st.number_input("pregnincies", 0,20,0)
glucose = st.number_input("glucose",0,300,200)
bp = st.number_input("blood pressure", 0, 200, 30)
skin = st.number_input("skin thickness", 0, 100, 50)
insulin = st.number_input("insulin", 0, 900, 300)
mi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 25)

input_data = np.array([[ preg, glucose, bp, skin, insulin, mi, dpf, age ]])

if st.button("predict"):
    scaled_data = scaler.transform(input_data) 
    prediction = model.predict(scaled_data)[0]
    proba = model.predict_proba(scaled_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ You are likely diabetic. Probability = {proba:.2f}")
    else:
        st.success(f"✅ You are not diabetic. Probability = {proba:.2f}")    