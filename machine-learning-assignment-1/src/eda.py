# eda.py

import pandas as pd

from src.utils import (
    get_categorical_cols,
    get_numeric_cols
)
from src.config import TARGET

def get_missing_values(df):
    missing_vals = df.isnull().sum().sort_values(ascending=False)
    missing_vals = missing_vals[missing_vals>0]
    return missing_vals

def check_duplicates(df):
    # duplicated rows
    return df.duplicated().sum()

def get_target_count(df):
    return df[TARGET].value_counts()

def target_distribution(df):
    # target ratio
    target_ratio =(
        (df[TARGET].value_counts(normalize=True)*100).round(2)
    ) 
    return target_ratio 

def numeric_summary(df):
    numeric_cols = get_numeric_cols(df)
    return df[numeric_cols].describe().T

def categorical_summary(df):
    categorical_cols = get_categorical_cols(df)
    return df[categorical_cols].describe().T

def get_memory_usage(df):
    return df.memory_usage(deep=True).sum() / 1024**2

def churn_rate_by_category(df, col):
    return (pd.crosstab(df[col],df[TARGET],normalize="index")*100).round(2)

def revenue_by_segment(df, col, revenue_col="MonthlyCharges"):
    return (
        df.groupby(col)[revenue_col]
        .agg(["mean", "sum", "count"])
        .sort_values(by="sum", ascending=False)
    )
