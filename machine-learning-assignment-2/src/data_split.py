# data_split.py

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE

def encode_target(y_train, y_validation, y_test):
    encoder = LabelEncoder()

    y_train_encoded = encoder.fit_transform(y_train)
    y_validation_encoded = encoder.transform(y_validation)
    y_test_encoded = encoder.transform(y_test)

    return (
        y_train_encoded,
        y_validation_encoded,
        y_test_encoded,
        encoder
    )

def split_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_val,
        y_train_val,
        test_size=VALIDATION_SIZE,
        stratify=y_train_val,
        random_state=RANDOM_STATE
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test
    )