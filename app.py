import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from preprocessing import preprocess_input

# Load model
model = joblib.load("model.pkl")

# Title
st.set_page_config(page_title="California House Price Predictor", page_icon="🏡")
st.title("🏡 California House Price Predictor")

# Load sample housing data to calculate ref values
@st.cache_data
def load_combined_data(folder="data"):
    dfs = []
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            dfs.append(pd.read_csv(os.path.join(folder, file)))
    return pd.concat(dfs, ignore_index=True)

sample_df = load_combined_data()
ref_values = {
    "LivingArea": sample_df["LivingArea"].median(),
    "LotSizeAcres": sample_df["LotSizeAcres"].median(),
    "HomeAge": 2025 - sample_df["YearBuilt"].median(),
    "City": sample_df["City"].mode()[0],
    "PostalCode": sample_df["PostalCode"].mode()[0],
    "PropertyType": sample_df["PropertyType"].mode()[0]
}

st.markdown("### ✍️ Enter Property Details")

with st.form("input_form"):
    LivingArea = st.number_input("Living Area (sq ft)", min_value=200, max_value=10000, value=1500)
    LotSizeAcres = st.number_input("Lot Size (acres)", min_value=0.01, max_value=10.0, value=0.25)
    YearBuilt = st.number_input("Year Built", min_value=1800, max_value=2025, value=1990)
    PropertyType = st.selectbox("Property Type", ["SingleFamilyResidence", "Condo", "Townhouse", "MultiFamily", "Other"])
    City = st.text_input("City", value="San Diego")
    PostalCode = st.text_input("Postal Code", value="92101")

    submitted = st.form_submit_button("Predict Price")

if submitted:
    try:
        input_df = pd.DataFrame({
            "LivingArea": [LivingArea],
            "LotSizeAcres": [LotSizeAcres],
            "YearBuilt": [YearBuilt],
            "PropertyType": [PropertyType],
            "City": [City],
            "PostalCode": [PostalCode]
        })

        processed_input = preprocess_input(input_df, ref_values)
        log_pred = model.predict(processed_input)[0]
        predicted_price = np.expm1(log_pred)

        st.markdown(f"### 💰 Predicted Home Price: **${predicted_price:,.0f}**")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
