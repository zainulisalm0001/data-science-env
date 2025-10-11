# house_prices/preprocess.py
# import numpy as np
# from sklearn.preprocessing import OneHotEncoder, StandardScaler

# def preprocess_train_data(X_train, cat_cols, num_cols):
#     """Fit and transform training data"""
#     encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
#     scaler = StandardScaler()

#     X_train_cat = encoder.fit(X_train[cat_cols]).transform(X_train[cat_cols])
#     X_train_num = scaler.fit(X_train[num_cols]).transform(X_train[num_cols])

#     X_train_processed = np.hstack([X_train_num, X_train_cat])

#     return X_train_processed, encoder, scaler

# def preprocess_inference_data(X, cat_cols, num_cols, encoder, scaler):
#     """Transform new data using fitted encoder and scaler"""
#     X_cat = encoder.transform(X[cat_cols])
#     X_num = scaler.transform(X[num_cols])
#     X_processed = np.hstack([X_num, X_cat])
#     return X_processed

import pandas as pd

def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering and preprocessing to input dataframe.

    Args:
        df (pd.DataFrame): Raw input dataframe

    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    df_processed = df.copy()

    # Example: add your preprocessing steps here
    # e.g., df_processed['NewFeature'] = df_processed['Feature'] * 2

    return df_processed