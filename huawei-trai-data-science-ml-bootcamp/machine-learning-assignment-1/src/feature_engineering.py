# feature_engineering.py

import numpy as np
import pandas as pd

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    featured["IsMonthToMonth"] = (featured["Contract"] == "Month-to-month").astype(int)
    featured["HasTechSupport"] = (featured["TechSupport"] == "Yes").astype(int)
    featured["IsNewCustomer"] = (featured["tenure"] <= 6).astype(int)
    featured["AverageMonthlySpend"] = featured["TotalCharges"] / featured["tenure"].replace(0, np.nan)
    return featured
