# modeling.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from xgboost import XGBClassifier

import pandas as pd

from src.config import RANDOM_STATE, CV_FOLDS
from src.preprocessing import build_model_pipeline
from src.utils import setup_logger

logger = setup_logger("modeling")

def get_models(): 
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            solver="saga",
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ),
        "decision_tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            max_depth = 6,
            learning_rate=0.01,
            random_state= RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss"
        )
    }

    return models

def get_param_grid(model_name):

    param_grids = {
        "logistic_regression": {
            "model__C": [0.1, 1.0, 10.0]
        },

        "decision_tree": {
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5, 10]
        },

        "random_forest": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5]
        },
        "xgboost": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 6],
            "model__learning_rate": [0.01, 0.1]
        }
    }

    if model_name not in param_grids:
        raise ValueError(
            f"No parameter grid defined for model: {model_name}"
        )

    return param_grids[model_name]

def compare_models(models, preprocessor, X_train, y_train, X_val, y_val): 
    # For each model, set up `build_model_pipeline(preprocessor, model)` and train it
    # Calculate precision, recall, and F1 score for validation (NOT accuracy, due to class imbalance)
    # Return the results as a DataFrame
    results = {}
    fitted_pipelines = {}

    for name, model in models.items():
        logger.info(f"Training {name}...")
        pipeline = build_model_pipeline(preprocessor, model)
        pipeline.fit(X_train,y_train)
        y_pred = pipeline.predict(X_val)

        results[name] = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred,average="weighted",zero_division=0),
            "recall": recall_score(y_val, y_pred,average="weighted",zero_division=0),
            "f1": f1_score(y_val, y_pred,average="weighted",zero_division=0),
        }
        fitted_pipelines[name] = pipeline
        logger.info(f"{name} — F1: {results[name]['f1']:.4f}")

    results_df = pd.DataFrame(results).T.sort_values("f1",ascending=False)
    return results_df, fitted_pipelines

def tune_best_model(pipeline, param_grid, X_train, y_train, scoring="f1_weighted"): 
    # GridSearchCV(pipeline, param_grid, scoring="f1", cv=CV_FOLDS)
    # Train and return best_estimator
    logger.info("Starting hyperparameter tuning...")
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=CV_FOLDS,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train,y_train)

    logger.info(f"Best params: {search.best_params_}")
    logger.info(f"Best CV F1: {search.best_score_:.4f}")

    return search.best_estimator_