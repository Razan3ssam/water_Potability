import streamlit as st
import numpy as np
import joblib
from keras.models import load_model

# Page Configuration

st.set_page_config(
    page_title="Water Potability Prediction",
    page_icon="💧",
    layout="wide"
)

# Load Model

model = load_model("water_potability.keras")
scaler = joblib.load("scaler (1).pkl")

# Sidebar

st.sidebar.title("Water Potability")
st.sidebar.markdown("""
### Neural Network Project

This application predicts whether water is **safe for drinking** using a trained Neural Network model.

**Model Inputs**
- pH
- Hardness
- Solids
- Chloramines
- Sulfate
- Conductivity
- Organic Carbon
- Trihalomethanes
- Turbidity
""")

# Main Title

st.title(" Water Potability Prediction using Neural Network")

st.markdown("---")

st.write("Enter the water characteristics then click **Predict**.")


# Input Section

col1, col2 = st.columns(2)

with col1:

    ph = st.number_input(
        "pH",
        min_value=0.0,
        max_value=14.0,
        value=7.0
    )

    hardness = st.number_input(
        "Hardness",
        value=200.0
    )

    solids = st.number_input(
        "Solids",
        value=15000.0
    )

    chloramines = st.number_input(
        "Chloramines",
        value=7.0
    )

    sulfate = st.number_input(
        "Sulfate",
        value=330.0
    )

with col2:

    conductivity = st.number_input(
        "Conductivity",
        value=420.0
    )

    organic_carbon = st.number_input(
        "Organic Carbon",
        value=14.0
    )

    trihalomethanes = st.number_input(
        "Trihalomethanes",
        value=65.0
    )

    turbidity = st.number_input(
        "Turbidity",
        value=4.0
    )

st.markdown("")

# Prediction


if st.button("Predict", use_container_width=True):

    sample = np.array([[
        ph,
        hardness,
        solids,
        chloramines,
        sulfate,
        conductivity,
        organic_carbon,
        trihalomethanes,
        turbidity
    ]])

    sample = scaler.transform(sample)

    prediction = model.predict(sample)

    probability = prediction[0][0]

    st.markdown("---")

    st.subheader("Prediction Result")

    st.progress(float(probability))

    if probability >= 0.5:

        st.success("Water is POTABLE (Safe to Drink)")

    else:

        st.error(" Water is NOT POTABLE")

    st.metric(
        label="Probability",
        value=f"{probability*100:.2f}%"
    )
