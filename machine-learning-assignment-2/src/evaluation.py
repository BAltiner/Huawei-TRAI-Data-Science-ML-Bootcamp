# evaluation.py

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import numpy as np
import pandas as pd

from src.utils import setup_logger

logger = setup_logger("evaluation")

def calculate_metrics(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    }

    return metrics

def evaluate_on_test(model, X_test, y_test):
    # predict
    # return calculate_metrics + classification_report + confusion matrix
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test,y_pred)
    report = classification_report(y_test,y_pred)
    matrix = confusion_matrix(y_test,y_pred)

    logger.info(f"Test metrrics: {metrics}")
    return {
        "metrics": metrics,
        "report": report,
        "confusion_matrix": matrix,
        "y_pred": y_pred
    }

def  analyze_errors(X_test, y_true, y_pred, group_col="proto"):
    # Filter out misclassified rows; analyze which type/protocol
    # has the most errors

    if group_col not in X_test.columns:
        raise ValueError(
            f"'{group_col}' is not available for error analysis."
        )
    
    errors = X_test.copy()
    errors["y_true"] = np.asarray(y_true)
    errors["y_pred"] = np.asarray(y_pred)
    errors["is_error"] = errors["y_true"] != errors["y_pred"]

    summary = (
        errors
        .groupby(group_col, dropna=False)
        .agg(
            sample_count=("is_error", "size"),
            error_count=("is_error", "sum"),
            error_rate=("is_error", "mean"),
        )
        .sort_values(
            by=["error_rate", "sample_count"],
            ascending=[False, False],
        )
    )

    logger.info(
        f"Error summary by '{group_col}':\n{summary.to_string()}"
    )

    return summary

def get_feature_importance(model, feature_names):
    # .feature_importances_ in the tree-based model, .coef_ in LogReg
    fitted_model = model.named_steps["model"]

    if hasattr(fitted_model, "feature_importances_"):
        values = fitted_model.feature_importances_

    elif hasattr(fitted_model, "coef_"):
        values = np.abs(fitted_model.coef_).ravel()

    else:
        raise ValueError(
            "Model does not expose feature_importances_ or coef_."
        )

    if "feature_selector" in model.named_steps:
        selector = model.named_steps["feature_selector"]
        feature_names = feature_names[selector.get_support()]

    importance = pd.Series(
        values,
        index=feature_names
    ).sort_values(ascending=False)

    return importance