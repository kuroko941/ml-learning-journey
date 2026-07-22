import streamlit as st
import joblib
import numpy as np

model = joblib.load("predictive_maintainance_model.pkl")
encoder = joblib.load("label_encoder.pkl")

st.title("Predictive Maintenance System")

machine_type = st.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)

air_temp = st.number_input(
    "Air Temperature (K)",
    value=298.0
)

process_temp = st.number_input(
    "Process Temperature (K)",
    value=308.0
)

rpm = st.number_input(
    "Rotational Speed (RPM)",
    value=1500
)

torque = st.number_input(
    "Torque (Nm)",
    value=40.0
)

tool_wear = st.number_input(
    "Tool Wear (min)",
    value=0
)

if st.button("Predict"):

    machine_type_encoded = encoder.transform([machine_type])[0]

    data = np.array([[
        machine_type_encoded,
        air_temp,
        process_temp,
        rpm,
        torque,
        tool_wear
    ]])

    prediction = model.predict(data)

    if prediction[0] == 0:
        st.success("Machine Healthy")
    else:
        st.error("Failure Risk Detected")
