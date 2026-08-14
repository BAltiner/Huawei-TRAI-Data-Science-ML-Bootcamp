# data_validation.py

from src.utils import setup_logger

logger = setup_logger("data_validation")
            
def validate_target(df, target_column):
    if target_column in df.columns:
        logger.info(f"Target column '{target_column}' is present in the dataset.")
        return True
    logger.error(f"Target column '{target_column}' is not in the dataset columns.")
    raise ValueError(f"Target column '{target_column}' is missing.")

def validate_duplicates(df):
    duplicated_count = df.duplicated().sum()
    if duplicated_count>0:
        logger.warning(f"Dataset contains {duplicated_count} duplicated rows.")
        return False
    logger.info("No duplicated rows found in the dataset.")
    return True

def validate_dataset(df, target_column):
    logger.info("Starting dataset validation pipeline...")

    try:
        validate_target(df, target_column)
        duplicates_valid = validate_duplicates(df)

        if not duplicates_valid:
            logger.warning(
                "Dataset validation completed with duplicated rows."
            )

        logger.info("Dataset validation pipeline completed successfully!")
        return True

    except ValueError as e:
        logger.critical(f"Dataset validation pipeline failed: {str(e)}")
        return False