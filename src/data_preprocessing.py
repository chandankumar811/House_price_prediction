import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def preprocess_data(train, test, target_col="SalePrice", log_target=True):
    # Drop Id column if it exists
    if "Id" in train.columns:
        train = train.drop(columns=["Id"])
    if "Id" in test.columns:
        test = test.drop(columns=["Id"])

    # Separate numeric and categorical
    num_cols = train.select_dtypes(include=["int64", "float64"]).columns.drop(target_col)
    cat_cols = train.select_dtypes(include=["object"]).columns

    # Fill missing values
    for col in num_cols:
         train.fillna({col: train[col].median()}, inplace=True)
         test.fillna({col: test[col].median()}, inplace=True)

    for col in cat_cols:
         train.fillna({col: "Missing"}, inplace=True)
         test.fillna({col: "Missing"}, inplace=True)

    # Features and target
    X = train.drop(columns=[target_col])
    y = train[target_col]

    # Log-transform target (optional but improves RMSE a lot!)
    if log_target:
        y = np.log1p(y)   # log(1 + SalePrice)

    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)
    test = pd.get_dummies(test, drop_first=True)

    X.to_csv("data/train_one_hot_encoded.csv")
    test.to_csv("data/test_one_hot_encoded.csv")


    # Align columns
    X, test = X.align(test, join="left", axis=1, fill_value=0)

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale numeric features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    test = scaler.transform(test)
    joblib.dump(scaler, "models/scaler.pkl")

    return X_train, X_val, y_train, y_val, test
