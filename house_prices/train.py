import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib


def build_model(data: pd.DataFrame) -> dict[str, float]:
    """
    Train model on training dataset and save it + preprocessing objects in models/ folder.
    Returns a dict with metrics (e.g. RMSE).
    """
    # Example categorical/numeric splits
    cat_cols = ["Neighborhood", "HouseStyle"]
    num_cols = ["LotArea", "OverallQual", "YearBuilt"]

    # Train/test split
    from sklearn.model_selection import train_test_split

    X = data[cat_cols + num_cols]
    y = data["SalePrice"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocessing
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    scaler = StandardScaler()

    X_train_cat = encoder.fit_transform(X_train[cat_cols])
    X_test_cat = encoder.transform(X_test[cat_cols])

    X_train_num = scaler.fit_transform(X_train[num_cols])
    X_test_num = scaler.transform(X_test[num_cols])

    X_train_processed = np.hstack([X_train_cat, X_train_num])
    X_test_processed = np.hstack([X_test_cat, X_test_num])

    # Model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train_processed, y_train)

    # Save objects
    joblib.dump(model, "models/model.joblib")
    joblib.dump(encoder, "models/encoder.joblib")
    joblib.dump(scaler, "models/scaler.joblib")

    # Evaluate
    preds = model.predict(X_test_processed)
    rmse = np.sqrt(((preds - y_test) ** 2).mean())

    return {"rmse": rmse}
