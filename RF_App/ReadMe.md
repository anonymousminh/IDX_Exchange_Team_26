# Real Estate Price Prediction App

This repository contains a Streamlit application for predicting housing prices using a trained Random Forest model and a custom preprocessing pipeline.

## Setup Instructions

1. Install required packages:

```
pip install -r requirements.txt
```

2. Run the training script to generate model files:

```
python train_model.py
```

This will create the following files in the current directory:
- model.pkl
- preprocessor.pkl
- price_bounds.pkl

3. Launch the Streamlit app:

```
streamlit run app.py
```

## File Structure

- app.py: Main Streamlit app
- train_model.py: Generates the model and preprocessor
- evaluate_model.py: Evaluates model performance
- preprocessing.py: Preprocessing pipeline class
- requirements.txt: Python package dependencies
- data/: Directory for input CSV files
