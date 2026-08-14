# utils.py

def get_categorical_cols(df):
    #  categorical columns
    categorical_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    return categorical_cols

def get_numeric_cols(df, target=None):
    # numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if target is not None and target in numeric_cols:
        numeric_cols.remove(target)
    return numeric_cols
