# data_loader.py

import pandas as pd
from src.config import DATA_PATH

# Data Loading
def load_data(file_path=DATA_PATH):
    """
    Load dataset from the DATA_PATH
    Returns dataframe
    """
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError("Dataset is empty.")
        print(f"Data loaded successfully. rows:{df.shape[0]}, columns:{df.shape[1]}")
        return df
    except (FileNotFoundError,ValueError, pd.errors.EmptyDataError) as error:
        print(f"Data loading failed: {error}")
        raise