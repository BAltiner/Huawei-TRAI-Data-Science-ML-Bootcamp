# data_loader.py

import pandas as pd

from src.config import DATA_FILE
from src.utils import setup_logger

logger = setup_logger("data_loader")

def load_data(file_path = None):
    if file_path is None:
        file_path = DATA_FILE

    logger.info(f"Starting read from CSV: {file_path}")

    try:
        df = pd.read_csv(file_path)

        if df.empty:
            logger.error(f"Dataset is empty..")
            raise ValueError("Dataset is empty.")
        
        logger.info(
            f"Data loaded successfully. rows={df.shape[0]}, columns={df.shape[1]}"
        )
        return df
    
    except (FileNotFoundError, pd.errors.EmptyDataError) as error:
        logger.error(f"Data loading failed: {error}.")
        raise
        