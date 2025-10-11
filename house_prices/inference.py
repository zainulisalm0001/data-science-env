# # house_prices/inference.py
# from house_prices.preprocess import preprocess_inference_data

# def make_predictions(input_data):
#     """
#     Load persisted model & preprocessing objects, transform data, and predict.
#     """
#     model = joblib.load("models/model.joblib")
#     encoder = joblib.load("models/encoder.joblib")
#     scaler = joblib.load("models/scaler.joblib")

#     # Identify columns
#     cat_cols = input_data.select_dtypes(include=['object']).columns.tolist()
#     num_cols = input_data.select_dtypes(include=['int64','float64']).columns.tolist()

#     # Preprocess and predict
#     X_processed = preprocess_inference_data(input_data, cat_cols, num_cols, encoder, scaler)
#     predictions = model.predict(X_processed)
#     return predictions


import pandas as pd
import numpy as np
import joblib


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    """
    Orchestrates the inference phase.

    Steps:
    - Load saved model from 'models/' folder
    - Preprocess input data
    - Make predictions

    Args:
        input_data (pd.DataFrame): Input dataframe for inference

    Returns:
        np.ndarray: Array of predicted house prices
    """
    # Load pre-trained objects
    model = joblib.load(
        "../models/model.joblib"
    )
    encoder = joblib.load(
        "../models/encoder.joblib"
    )
    scaler = joblib.load(
        "../models/scaler.joblib"
    )

    # Preprocess input (replace with your real preprocessing)
    input_encoded = encoder.transform(
        input_data
    )
    input_scaled = scaler.transform(
        input_encoded
    )

    # Make predictions
    predictions = model.predict(
        input_scaled
    )

    return predictions