import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
price_bounds = joblib.load("price_bounds.pkl")

st.title("🏠 California Home Price Estimator")
st.markdown("Input the home's features below:")

# ----------- User Inputs -----------
sqft = st.number_input("Living Area (sqft)", 500.0, 10000.0, value=2000.0)
baths = st.number_input("Bathrooms", 0, 10, step=1, value=2)
beds = st.number_input("Bedrooms", 0, 10, step=1, value=3)
lot_size = st.number_input("Lot Size (sqft)", 0.0, 50000.0, value=5000.0)
garage = st.number_input("Garage Spaces (The number of spaces in the garages: decimal)", 0.0, 20.0, value=2.0)
parking = st.number_input("Total Parking Spaces (integer)", 0.0, 50.0, step=1.0, value=2.0)
year_built = st.number_input("Year Built", 1700.0, 2025.0, value=2000.0)
assoc_fee = st.number_input("Association Fee ($/mo)", 0.0, 3000.0, value=0.0)
main_level_bed = st.number_input("Main Level Bedrooms", 0.0, 5.0, step=1.0, value=2.0)
levels = st.selectbox("Levels", ["One", "Two", "ThreeOrMore", "MultiSplit"])
flooring = st.multiselect(
    "Flooring Materials", 
    ['Carpet', 'Tile', 'Wood', 'Laminate', 'Vinyl', 'Stone', 'Concrete', 'Bamboo'], 
    default=['Wood']
)

# Binary flags
fireplace = st.checkbox("Fireplace", value=True)
pool = st.checkbox("Private Pool", value=False)
view = st.checkbox("Scenic View", value=False)

# Location info
zipcode = st.text_input("ZIP Code", "90210")
city = st.text_input("City", "Beverly Hills")
county = st.text_input("County/Parish", "Los Angeles")
district = st.text_input("High School District", "Los Angeles Unified")
lat = st.number_input("Latitude", 0.0, 120.0, value=34.09)
lon = st.number_input("Longitude", -180.0, 0.0, value=-118.41)

# ----------- Prediction Logic -----------
if st.button("💰 Predict Home Price"):
    flooring_str = ", ".join(flooring) if flooring else ""
    raw_input = pd.DataFrame([{
        "LivingArea": sqft,
        "BathroomsTotalInteger": baths,
        "BedroomsTotal": beds,
        "LotSizeSquareFeet": lot_size,
        "GarageSpaces": garage,
        "ParkingTotal": parking,
        "YearBuilt": year_built,
        "AssociationFee": assoc_fee,
        "MainLevelBedrooms": main_level_bed,
        "Levels": levels,
        "FireplaceYN": int(fireplace),
        "PoolPrivateYN": int(pool),
        "ViewYN": int(view),
        "Flooring": flooring_str,
        "PostalCode": zipcode,
        "City": city,
        "CountyOrParish": county,
        "HighSchoolDistrict": district,
        "Latitude": lat,
        "Longitude": lon,
        "PropertyType": "Residential",
        "PropertySubType": "SingleFamilyResidence"
    }])

    processed = preprocessor.transform(raw_input)
    X_input = processed[model.feature_names_in_]
    log_pred = model.predict(X_input)[0]
    price = np.expm1(log_pred)
    st.success(f"Estimated Home Price: **${price:,.0f}**")