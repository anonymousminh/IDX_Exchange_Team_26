
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from preprocessing import RealEstatePreprocessor
from sklearn.base import BaseEstimator, TransformerMixin
def apply_price_filter_by_quantile(df, lower_quantile=0.01, upper_quantile=0.99, verbose=True):
    '''cutting extreme value beyond give percentile'''
    q_low = df['ClosePrice'].quantile(lower_quantile)
    q_high = df['ClosePrice'].quantile(upper_quantile)
    mask = df['ClosePrice'].between(q_low, q_high)
    if verbose:
        print(f"Applied quantile filter: [{q_low:.0f}, {q_high:.0f}]")
        print(f"Filtered out {(~mask).sum()} samples ({(~mask).mean():.2%})")
    return df[mask].copy(), (q_low, q_high)
# for training
features = ['LivingArea', 'BathroomsTotalInteger', 'LatBin', 'LatLonCluster', 'LonBin',
            'PostalCodeFreq', 'CountyOrParishFreq', 'CityFreq', 'LotSizeSquareFeet', 'HasWood',
            'YearBuilt', 'HighSchoolDistrictFreq', 'AssociationFee', 'ParkingTotal',
            'BedroomsTotal', 'PoolPrivateYN', 'GarageSpaces', 'MainLevelBedrooms',
            'Levels', 'FireplaceYN', 'ViewYN', 'HasCarpet']

# file contains all six months data, we already filtered by property type and sub type, and dropped some meaningless columns
train_raw_df = pd.read_csv("data/filtered_df_dropped.csv") 
preprocessor = RealEstatePreprocessor(final_features=features, enforce_property_type=False)
train_processed = preprocessor.fit_transform(train_raw_df)
train_filtered, price_bounds = apply_price_filter_by_quantile(train_processed)

# save the price bounds for later use
joblib.dump(price_bounds, "price_bounds.pkl")

X_train = train_filtered[features]
y_train = np.log1p(train_filtered['ClosePrice'])

final_model = RandomForestRegressor(max_depth=20, n_estimators=200, random_state=42)
final_model.fit(X_train, y_train)

# save the model and preprocessor
joblib.dump(final_model, "model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")
