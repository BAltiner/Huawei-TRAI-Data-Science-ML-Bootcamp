# evaluation.py

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def find_best_threshold(model, X_val, y_val):
    proba = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, proba)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores[:-1])
    return thresholds[best_idx], f1_scores[best_idx]

# Validation
def evaluate_validation(model, X_val, y_val):
    # Calculate
    y_pred = model.predict(X_val)
    # Accuracy-> genel doğruluk
    model_acc = accuracy_score(y_val, y_pred)
    # Precision -> Churn olacak sonucun ne kadarı gerçekten churn?
    model_precision = precision_score(y_val, y_pred,zero_division=0)
    # Recall -> Gerçek churn müşterilerinin ne kadarını yakalandı
    model_recall = recall_score(y_val, y_pred,zero_division=0)
    # F1
    model_f1 = f1_score(y_val, y_pred,zero_division=0)
    # Return comparison table.
    conf_table = confusion_matrix(y_val, y_pred)
    return {"accuracy":model_acc,
            "precision":model_precision,
            "recall":model_recall,
            "f1":model_f1,
            "confusion_matrix":conf_table}


# Final Test Evaluation
def evaluate_models_on_test(best_model, X_test, y_test, threshold=0.5):
    proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    # Accuracy
    model_acc = accuracy_score(y_test,y_pred)
    # Precision
    model_precision = precision_score(y_test,y_pred, zero_division=0)
    # Recall
    model_recall = recall_score(y_test,y_pred, zero_division=0)
    # F1
    model_f1 = f1_score(y_test,y_pred, zero_division=0)
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test,y_pred)
    plt.figure()
    sns.heatmap(conf_matrix,annot=True, fmt="d",cmap="Blues",xticklabels=["No Churn", "Churn"],yticklabels=["No Churn", "Churn"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {best_model.__class__.__name__} (thr={threshold:.3f})")
    plt.tight_layout()
    plt.show()
    # Classification Report
    model_report = classification_report(y_test,y_pred)
    return {"accuracy":model_acc,
            "precision":model_precision,
            "recall":model_recall,
            "f1":model_f1,
            "confusion_matrix":conf_matrix,
            "classification_report": model_report}