# visualization.py

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import shap

sns.set_style("whitegrid",rc={"axes.facecolor": "#f8f9fac7"})

def plot_class_distribution(df, target_col): 
    plt.figure(figsize=(5,4))
    sns.countplot(df,x=target_col)
    plt.title("Target Distribution")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
  
def plot_numeric_distribution(df,numeric_cols,target_col, log_scale=False):
    cols_to_plot = [col for col in numeric_cols if col != target_col]

    n_cols = 3
    n_rows = math.ceil(len(cols_to_plot)/n_cols)

    fig, axes = plt.subplots(
        n_rows, 
        n_cols, 
        figsize=(15, n_rows * 4)
    )
    axes = np.atleast_1d(axes).flatten()

    for i, col in enumerate(cols_to_plot):
        plot_df = df[[col, target_col]].dropna().copy()
        
        if log_scale:
            if (plot_df[col] < 0).any():
                raise ValueError(
                    f"{col} contains negative values; log1p cannot be used."
                )

            plot_df[col] = np.log1p(plot_df[col])

        sns.histplot(
            data=plot_df,
            x=col,
            hue=target_col,
            stat="density",
            common_norm=False, #Each class is normalized according to its own distribution.
            multiple="layer",
            element="step",
            fill=False,
            bins=40,
            ax=axes[i],
        )

        suffix = " (log1p)" if log_scale else ""
        axes[i].set_title(f"{col}{suffix}")
        axes[i].set_ylabel("Density")

    for i in range(len(cols_to_plot), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

def plot_outliers(df, numeric_columns, log_scale=False):
    n_cols = 3
    n_rows = math.ceil(len(numeric_columns) / n_cols)
    fig, axes = plt.subplots(
        n_rows, 
        n_cols, 
        figsize=(15, n_rows*5)
    )
    axes = axes.flatten()

    for i,col in enumerate(numeric_columns):
        data = df[col].dropna()
        if log_scale:
            data = np.log1p(data)
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3- q1

        axes[i].boxplot(data)
        axes[i].set_title(f"{col}{' (log1p)' if log_scale else ''}\n(IQR: {iqr:.2f})")

    for i in range(len(numeric_columns), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df, numeric_cols):
    corr = df[numeric_cols].corr()
    size = max(6, len(numeric_cols) * 0.6)

    plt.figure(figsize=(size, size * 0.8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        annot_kws={"size": 8}
    )
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name,
    labels,
    display_labels,
    normalize=None,
):
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        normalize=normalize,
    )

    fmt = ".2f" if normalize else "d"

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=display_labels,
        yticklabels=display_labels,
    )

    plt.title(f"Confusion Matrix — {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

def plot_model_comparison(results): 
    results.plot(kind="bar",figsize=(9,5))
    plt.title("Model Comparison — Validation Metrics")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

def plot_shap_summary(pipeline, X_test, sample_size=1000, random_state=None):
    X_shap = X_test.sample(
        n=min(sample_size, len(X_test)),
        random_state=random_state,
    )

    preprocessor = pipeline.named_steps["preprocessor"]
    feature_selector = pipeline.named_steps["feature_selector"]
    model = pipeline.named_steps["model"]

    X_shap_transformed = preprocessor.transform(X_shap)
    X_shap_selected = feature_selector.transform(X_shap_transformed)

    if hasattr(X_shap_selected, "toarray"):
        X_shap_selected = X_shap_selected.toarray()

    feature_names = preprocessor.get_feature_names_out()
    selected_mask = feature_selector.get_support()
    selected_feature_names = feature_names[selected_mask]

    X_shap_df = pd.DataFrame(X_shap_selected, columns=selected_feature_names)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap_df, check_additivity=False)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    shap.summary_plot(shap_values, X_shap_df, show=True)