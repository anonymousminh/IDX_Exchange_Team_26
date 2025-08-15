import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# load the model and preprocessor
model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
price_bounds = joblib.load("price_bounds.pkl")

# Define the features used in training
features = ['LivingArea', 'BathroomsTotalInteger', 'LatBin', 'LatLonCluster', 'LonBin',
            'PostalCodeFreq', 'CountyOrParishFreq', 'CityFreq', 'LotSizeSquareFeet', 'HasWood',
            'YearBuilt', 'HighSchoolDistrictFreq', 'AssociationFee', 'ParkingTotal',
            'BedroomsTotal', 'PoolPrivateYN', 'GarageSpaces', 'MainLevelBedrooms',
            'Levels', 'FireplaceYN', 'ViewYN', 'HasCarpet']

# Load the test data
test_raw_df = pd.read_csv("data/CRMLSSold202506.csv")

# Apply the preprocessor
test_processed = preprocessor.transform(test_raw_df)

# Filter extreme prices
test_filtered = test_processed[
    test_processed['ClosePrice'].between(*price_bounds)
].copy()

# Prepare the test features
X_test = test_filtered[features]
y_actual = test_filtered['ClosePrice']

# Model prediction (log → actual price)
y_pred = model.predict(X_test)
y_pred_actual = np.expm1(y_pred)

# Evaluation metrics
mape = mean_absolute_percentage_error(y_actual, y_pred_actual)
r2 = r2_score(y_actual, y_pred_actual)

# Print results
print(f"Test MAPE: {mape:.2%}")
print(f"R² Score: {r2:.3f}")
