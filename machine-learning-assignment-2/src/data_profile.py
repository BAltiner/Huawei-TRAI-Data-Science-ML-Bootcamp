# data_profile.py

import pandas as pd

from src.utils import setup_logger
from src.preprocessing import (
    get_numeric_columns,
    get_categoric_columns
)

logger = setup_logger("data_profile")

def get_missing_summary(df): 
    missings = df.isna()
    missing_rows = df[missings.any(axis=1)]
    missing_row_count = len(missing_rows)
    logger.info(f"Total rows with missing values: {missing_row_count}")
    return {
        "missing row count": missing_row_count,
        "missing rows": missing_rows
    }

def get_duplicate_summary(df):
    duplicates = df.duplicated(keep=False)
    duplicated_rows = df[duplicates]
    duplicated_row_count = len(duplicated_rows)
    logger.info(f"Duplicated row count: {duplicated_row_count}")
    return {
        "duplicated row count": duplicated_row_count,
        "duplicated rows": duplicated_rows
    }

def get_target_distribution(df, target_column):
    distribution  = df[target_column].value_counts().sort_values(ascending=False)
    return distribution

def get_target_rate(df,target_column):
    target_rate = df[target_column].value_counts(normalize=True) * 100
    return target_rate

def get_numeric_summary(df): 
    numeric_cols = get_numeric_columns(df)
    if not numeric_cols:
        return pd.DataFrame()
    return df[numeric_cols].describe().T

def get_categorical_summary(df):
    categoric_cols = get_categoric_columns(df)
    return df[categoric_cols].describe().T

def get_cardinality_summary(df):
    categoric_cols = get_categoric_columns(df)
    if not categoric_cols:
        return pd.Series(dtype=int)
    cardinality = {col: df[col].nunique() for col in categoric_cols}
    return pd.Series(cardinality).sort_values(ascending=False)