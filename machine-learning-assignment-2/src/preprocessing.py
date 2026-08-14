# preprocessing.py

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from src.config import RANDOM_STATE
from src.utils import setup_logger

logger = setup_logger("preprocessing")

def get_numeric_columns(df):
    return df.select_dtypes(include=["number"]).columns.to_list()

def get_categoric_columns(df):
    return df.select_dtypes(include=["object","string","category"]).columns.to_list()

def strip_strings(df):
    categorical_cols = get_categoric_columns(df)
    for col in categorical_cols:
        df[col] = df[col].str.strip()
    return df

def replace_missing_markers(df):
    return df.replace("-",np.nan)

def remove_exact_duplicates(df):
    before = len(df)
    deduplicated_df = df.drop_duplicates().copy()
    after = len(deduplicated_df)
    removed_count = before - after
    
    logger.info(
        f"Removed exact duplicates: {removed_count}. "
        f"Rows remaining: {len(deduplicated_df)}."
    )

    return deduplicated_df

def get_near_empty_columns(df, threshold = 0.95):
    missing_ratio = df.isna().mean()
    return missing_ratio[missing_ratio>threshold].index.to_list()

def get_zero_variance_columns(df):
    return [col for col in df.columns if df[col].nunique(dropna=True) <=1]

def build_preprocessor(numerical_columns, categorical_columns):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])

    categoric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown ="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline,numerical_columns),
        ("categoric", categoric_pipeline, categorical_columns)
    ])

    return preprocessor

def build_model_pipeline(preprocessor, model):
    selector = SelectFromModel(
        estimator=RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        threshold="median"
    )

    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("feature_selector", selector),
        ("model", model)
    ])
    return model_pipeline