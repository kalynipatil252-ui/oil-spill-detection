import streamlit as st
import joblib
import numpy as np

model = joblib.load("oil_spill_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("AI-Based Oil Spill Detection System")

st.write("Enter satellite sensor feature values:")

inputs = []
for i in range(1, 50):
    val = st.number_input(f"Feature {i}", value=0.0)
    inputs.append(val)

if st.button("Predict"):
    data = np.array(inputs).reshape(1, -1)
    data = scaler.transform(data)
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("⚠️ Oil Spill Detected")
    else:
        st.success("✅ No Oil Spill Detected")
