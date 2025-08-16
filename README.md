# California Property Price Predictor

A comprehensive machine learning project that predicts California residential property prices using advanced data preprocessing, feature engineering, and ensemble modeling techniques.

## Project Overview

This project analyzes California Multiple Listing Service (MLS) data to build predictive models for residential property prices. The system processes raw real estate data through sophisticated preprocessing pipelines and employs multiple machine learning algorithms to achieve accurate price predictions.

## Key Features

- **Interactive Web Application**: Streamlit-based interface for easy property price predictions
- **Advanced Data Preprocessing**: Automated cleaning, feature engineering, and data validation
- **Multiple ML Models**: KNN, Random Forest, and LightGBM implementations with hyperparameter optimization
- **Geospatial Analysis**: Latitude/longitude clustering and binning for location-based features
- **Comprehensive Feature Engineering**: Handles missing data, categorical encoding, and outlier detection
- **Model Persistence**: Saved models for production deployment

## Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd IDX_Exchange_Team_26
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the web application:
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Project Structure

```
IDX_Exchange_Team_26/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── model.pkl                      # Trained machine learning model
├── cleaned_data.csv               # Preprocessed dataset
├── raw_data.csv                   # Original raw dataset
├── data/                          # Monthly MLS data files
│   ├── CRMLSSold202412.csv
│   ├── CRMLSSold202501_filled.csv
│   ├── CRMLSSold202502.csv
│   ├── CRMLSSold202503.csv
│   ├── CRMLSSold202504.csv
│   ├── CRMLSSold202505.csv
│   └── CRMLSSold202506.csv
├── notebooks/                     # Jupyter notebooks for analysis
│   ├── 00_extract.ipynb         # Data extraction and loading
│   ├── 01_preprocessing.ipynb   # Data cleaning and preprocessing
│   ├── 02_modeling.ipynb        # Model training and evaluation
│   ├── RF_00_SlidingWindow&GridSearch.ipynb  # Random Forest optimization
│   └── RF_V2.ipynb              # Random Forest version 2
└── scripts/                       # Python scripts for automation
    ├── RF_App_01_preprocessing.py  # Automated preprocessing pipeline
    └── RF_App_02_train_model.py   # Model training automation
```

## Technical Implementation

### Data Preprocessing Pipeline

The project implements a sophisticated preprocessing pipeline (`RealEstatePreprocessor`) that:

- Filters for residential single-family properties
- Handles missing data with intelligent imputation strategies
- Creates geospatial features (latitude/longitude clustering)
- Normalizes categorical variables
- Applies outlier capping for numerical features
- Maintains preprocessing state for consistent transformations

### Machine Learning Models

1. **K-Nearest Neighbors (KNN)**
   - Numerical features only: MAPE: 0.430, R²: 0.317
   - With categorical features: Enhanced performance through feature engineering

2. **Random Forest**
   - Grid search optimization with sliding window validation
   - Hyperparameter tuning using Optuna
   - Feature importance analysis

3. **LightGBM**
   - Gradient boosting with categorical feature support
   - Optimized hyperparameters for real estate data

### Feature Engineering

- **Geospatial Features**: Latitude/longitude clustering and binning
- **Temporal Features**: Property age calculations
- **Categorical Encoding**: One-hot encoding with frequency-based imputation
- **Numerical Transformations**: Log transformations for skewed distributions
- **Outlier Handling**: 99th percentile capping for extreme values

## Data Sources

The project utilizes California MLS data including:
- Property characteristics (living area, lot size, bedrooms, bathrooms)
- Location data (city, postal code, latitude, longitude)
- Property details (year built, property type, garage spaces)
- Market information (days on market, association fees)

## Usage Examples

### Web Application

1. Open the Streamlit app
2. Enter property details:
   - Living area (sq ft)
   - Lot size (acres)
   - Year built
   - Property type
   - City and postal code
3. Click "Predict Price" to get instant price estimates

### Programmatic Usage

```python
from preprocessing import preprocess_input
import joblib

# Load trained model
model = joblib.load("model.pkl")

# Prepare input data
input_data = pd.DataFrame({
    "LivingArea": [1500],
    "LotSizeAcres": [0.25],
    "YearBuilt": [1990],
    "PropertyType": ["Residential"],
    "City": ["San Diego"],
    "PostalCode": ["92101"]
})

# Make prediction
prediction = model.predict(preprocess_input(input_data))
```

## Model Performance

- **KNN Model**: MAPE: 0.430, R²: 0.317
- **Random Forest**: Enhanced performance through feature engineering
- **LightGBM**: Optimized for categorical and numerical features

## Key Insights

The analysis reveals several important factors affecting California property prices:
- Location (city, school district) significantly impacts pricing
- Property size and age are strong predictors
- Geospatial clustering improves model accuracy
- Categorical features require careful encoding strategies

## Dependencies

- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, lightgbm
- **Optimization**: optuna
- **Web Framework**: streamlit
- **Model Persistence**: joblib

## Contributing

This project was developed by IDX Exchange Team 26. For questions or contributions, please contact the team.

## License

This project is for educational and research purposes.

## Important Notes

- The model is trained on California MLS data and may not generalize to other regions
- Property prices are predictions and should not be used as the sole basis for financial decisions
- Regular model retraining is recommended as market conditions change

---

**Built with <3 by IDX Exchange Team 26**
