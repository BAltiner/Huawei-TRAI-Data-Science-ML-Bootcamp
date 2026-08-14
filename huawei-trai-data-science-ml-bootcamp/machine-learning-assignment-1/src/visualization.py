# visualization.py

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid",rc={"axes.facecolor": "#f8f9fac7"})

from src.eda import get_missing_values
from src.utils import get_numeric_cols

def plot_boxplots(df):
    numeric_cols = get_numeric_cols(df)
    for col in numeric_cols:
        plt.figure(figsize=(5,4))
        sns.boxplot(x=df[col])
        plt.title(col)
        plt.tight_layout()
        plt.show()


def plot_missing_values(df):
    missing_values = get_missing_values(df)
    if missing_values.empty:
        print("There is no empty value..")
        return
    plt.figure(figsize=(5,4))
    missing_values.plot(kind="bar")
    plt.ylabel("Missing Values")
    plt.tight_layout()
    plt.show()

def plot_target_distribution(df):
    plt.figure(figsize=(5,4))
    sns.countplot(df, x="Churn")
    plt.title("Target Distribution")
    plt.tight_layout()
    plt.show()


def plot_numeric_distribution(df):
    numeric_cols = get_numeric_cols(df)
    for col in numeric_cols:
        plt.figure(figsize=(5,4))
        sns.histplot(df[col], kde=True)
        plt.title(col)
        plt.tight_layout()
        plt.show()

def plot_correlation_heatmap(df):
    numeric_cols = get_numeric_cols(df)
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(5,4))
    sns.heatmap(corr,
                annot=True,
                cmap="coolwarm",
                fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, model_name):
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No churn", "Churn"],
        yticklabels=["No churn", "Churn"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.show()
