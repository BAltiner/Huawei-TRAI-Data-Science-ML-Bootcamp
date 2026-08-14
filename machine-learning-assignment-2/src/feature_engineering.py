# feature_engineering.py

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from src.utils import setup_logger
from src.preprocessing import(
    get_near_empty_columns,
    get_zero_variance_columns
)
from src.config import RANDOM_STATE, IDENTIFIER_COLUMNS

logger = setup_logger("feature_engineering")

def drop_leakage_columns(df,leakage_columns):
    cols_to_drop = [col for col in leakage_columns if col in df.columns]
    logger.info(f"Dropping leakage columns: {cols_to_drop}")

    return df.drop(columns= cols_to_drop)

def drop_identifier_columns(df):
    cols_to_drop = [col for col in IDENTIFIER_COLUMNS if col in df.columns]
    logger.info(f"Dropping identifier columns: {cols_to_drop}")

    return df.drop(columns=cols_to_drop)

def create_presence_indicators(df):
    featured = df.copy()
    # leakage/yüksek kardinaliteli sütunları çıkar (IP'ler dahil)

    ssl_cols = [col for col in df.columns if col.startswith("ssl_")]
    http_cols = [col for col in df.columns if col.startswith("http_")]

    if ssl_cols:
        featured["has_ssl"] = (
            featured[ssl_cols]
            .notna()
            .any(axis=1)
            .astype("int8")
        )
        featured = featured.drop(columns=ssl_cols)
        logger.info(f"Created 'has_ssl' from {ssl_cols}")

    if http_cols:
        http_numeric_cols = [
            "http_request_body_len",
            "http_response_body_len",
            "http_status_code",
        ]

        http_text_cols = [
            col for col in http_cols
            if col not in http_numeric_cols
        ]

        http_is_present = (
            featured[http_text_cols].notna().any(axis=1)
            | featured[http_numeric_cols]
            .fillna(0)
            .ne(0)
            .any(axis=1)
        )

        featured["has_http"] = http_is_present.astype("int8")
        featured = featured.drop(columns=http_cols)

        logger.info(
            "Created 'has_http' using non-empty HTTP text fields "
            "or non-zero HTTP numeric fields."
        )

    return featured

def drop_zero_variance_columns(df):
    zero_var_cols = get_zero_variance_columns(df)
    if zero_var_cols:
        logger.info(f"Dropping zero-variance columns: {zero_var_cols}")
        return df.drop(columns=zero_var_cols)
    return df

def create_ratio_features(df):
    featured = df.copy()
    featured["bytes_ratio"] = df["src_bytes"] / (df["dst_bytes"] + 1)
    featured["pkts_ratio"] = df["src_pkts"] / (df["dst_pkts"] + 1)
    logger.info("Created ratio features: bytes_ratio, pkts_ratio")
    return featured

def create_features(df: pd.DataFrame, leakage_columns: list) -> pd.DataFrame:
    featured = df.copy()
    featured = drop_leakage_columns(
        featured,
        leakage_columns
    )
    featured = drop_identifier_columns(featured)
    featured = create_presence_indicators(featured)
    featured = drop_zero_variance_columns(featured)
    featured = create_ratio_features(featured)

    logger.info(f"Feature engineering completed. Final shape: {featured.shape}")
    return featured