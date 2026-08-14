# preprocessing.py

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from src.config import TARGET, TEST_SIZE,RANDOM_STATE, VALIDATION_SIZE
from src.utils import get_categorical_cols, get_numeric_cols
from src.eda import check_duplicates

def remove_duplicates(df):
    duplicates = check_duplicates(df)
    if duplicates:
        df = df.drop_duplicates()
    return df

def strip_strings(df):
    categorical_cols = get_categorical_cols(df)
    for col in categorical_cols:
        df[col] = df[col].str.strip()
    return df

def replace_blank_with_nan(df):
    return df.replace(r"^\s*$", np.nan, regex=True)

def fill_missing_values(df):
    numeric_cols = get_numeric_cols(df)
    categorical_cols = get_categorical_cols(df)
    try:
        for col in numeric_cols:
            col_mean = df[col].mean()
            df[col] = df[col].fillna(col_mean)
        for col in categorical_cols:
            modes = df[col].mode()
            if not modes.empty:
                col_mode = modes.iloc[0]
                df[col] = df[col].fillna(col_mode)
    except Exception as e:
        print(f"An error occurred while filling missing values: {e}")
    return df

def remove_invalid_values(df):pass

def drop_columns(df, columns):
    return df.drop(columns=[c for c in columns if c in df.columns])

def convert_numeric_columns(df, columns):
    if isinstance(columns, str):
        columns = [columns]

    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def encode_target(df):
    if TARGET not in df.columns:
        raise KeyError(f"{TARGET} column can not found..")

    target_series = df[TARGET].astype("string").str.strip()
    target_series = target_series.map({
        "No":0,
        "Yes":1
    })
    if target_series.isna().any():
        mode_value = target_series.mode()[0]
        target_series = target_series.fillna(mode_value)
    df[TARGET] = target_series.astype(int)
    return df
    
# Data Cleaning
def clean_data(df): 
    # duplicated rows
    df_clean = df.copy()
    df_clean = remove_duplicates(df_clean)
    df_clean = strip_strings(df_clean)
    df_clean = replace_blank_with_nan(df_clean)
    # df_clean = fill_missing_values(df_clean)

    return df_clean

# Data - Target Split
def data_target_split(df):
    try:
        if TARGET not in df.columns:
            raise KeyError("The target column 'Churn' is not present in the DataFrame.")
        y= df[TARGET]
        X = df.drop(columns=[TARGET])
        return X, y
    except KeyError as e:
        print(f"Error in data_target_split: {e}")
        print(f"Available columns: {df.columns}")
        return None, None

# Train / Validation / Test Split
def train_val_test_split(X,y):
    try:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size = TEST_SIZE,
            random_state= RANDOM_STATE,
            stratify= y # stratify ile hedef sütundaki churn oranını korundu
        )

        # %60 train, %20 validation, %20 test
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size= VALIDATION_SIZE,
            random_state= RANDOM_STATE,
            stratify=y_train_val
        )

        return (X_train, X_val, X_test, y_train, y_val, y_test)
    except Exception as e:
        print(f"Error in train_val_test_split: {e}")
        raise

# Preprocessing Pipeline
def build_preprocessor(numerical_features, categorical_features):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categoric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numerical_features),
        ("categoric", categoric_pipeline, categorical_features)
    ])
    return preprocessor