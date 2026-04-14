"""
preprocess.py
-------------
Loads retail_sales.csv, cleans it, and engineers features.
"""

import pandas as pd
import numpy as np


def load_and_preprocess(filepath="data/retail_sales.csv"):
    df = pd.read_csv(filepath, parse_dates=["Date"])

    # Sort by date
    df = df.sort_values("Date").reset_index(drop=True)

    # Fill missing values
    df = df.ffill().bfill()

    # Feature Engineering
    df["year"]       = df["Date"].dt.year
    df["month"]      = df["Date"].dt.month
    df["day"]        = df["Date"].dt.day
    df["dayofweek"]  = df["Date"].dt.dayofweek
    df["quarter"]    = df["Date"].dt.quarter
    df["is_weekend"] = (df["Date"].dt.dayofweek >= 5).astype(int)
    df["week"]       = df["Date"].dt.isocalendar().week.astype(int)

    # Encode categorical columns
    df["Store_Code"]   = df["Store"].astype("category").cat.codes
    df["Product_Code"] = df["Product"].astype("category").cat.codes

    print(f"✅ Preprocessed: {len(df):,} rows | {df.shape[1]} columns")
    return df


def get_features_target(df):
    feature_cols = [
        "month", "day", "dayofweek", "quarter",
        "is_weekend", "week", "Store_Code",
        "Product_Code", "Price"
    ]
    X = df[feature_cols]
    y = df["Sales"]
    return X, y


if __name__ == "__main__":
    df = load_and_preprocess()
    X, y = get_features_target(df)
    print(f"Features : {list(X.columns)}")
    print(f"X shape  : {X.shape}")
    print(df.head())