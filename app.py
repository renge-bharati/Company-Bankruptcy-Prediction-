import streamlit as st
import numpy as np
import pickle

# ===============================
# LOAD MODEL
# ===============================
with open("bankruptcy_rf_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Company Bankruptcy Prediction", layout="wide")

st.title("🏢 Company Bankruptcy Prediction App")
st.write("Predict whether a company is likely to go bankrupt using financial indicators.")

# ===============================
# USER INPUTS
# ===============================
st.sidebar.header("Enter Financial Values")

feature_names = model.feature_names_in_

inputs = []
for feature in feature_names:
    value = st.sidebar.number_input(f"{feature}", value=0.0)
    inputs.append(value)

# ===============================
# PREDICTION
# ===============================
if st.button("Predict Bankruptcy"):
    input_array = np.array(inputs).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Bankruptcy\n\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Company is Financially Safe\n\nProbability: {probability:.2f}")
