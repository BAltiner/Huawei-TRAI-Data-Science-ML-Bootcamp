#  modeling.py

# Model Definitions
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE

def build_models():
    models = {
        # Logistic Regression
        "Logistic Regression" : LogisticRegression(random_state=RANDOM_STATE, max_iter=1000,class_weight="balanced"),

        # KNN
        "KNN" : KNeighborsClassifier(n_neighbors=5),

        # Decision Tree (Bonus)
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE)
    }
    return models

# Training
def train_models(model, X_train, y_train):
    return model.fit(X_train, y_train)

# Best Model Selection
def select_best_model(validation_results):
    best_model = max(validation_results,
                     key= lambda x: x["f1"])
    return best_model