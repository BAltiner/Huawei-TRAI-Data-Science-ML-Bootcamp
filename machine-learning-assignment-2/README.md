# TON-IoT Network Intrusion Detection

## Project Overview

In this project, TON-IoT network data was used to classify whether network traffic contains an attack, and if so, which attack type it belongs to.

The project is handled in two stages:

1. **Binary Classification:** Is the traffic `normal` or an `attack`?
2. **Attack Classification:** If the traffic is an attack, which attack type does it belong to?

The goal is not only to achieve high accuracy but also to analyze the data structure to engineer meaningful features, prevent data leakage, and establish a reliable intrusion detection pipeline by comparing model behaviors.

---

## 1. Data Understanding

In the first stage, the structure of the dataset was analyzed.

Missing values, duplicate records, numeric/categorical variables, cardinality, target distributions, and the distributions of numeric variables were examined.

Unnecessary whitespaces in string fields were cleaned, and `-` values in the dataset were treated as missing values.

Duplicate records were analyzed and included in the data cleaning process.

---

## 2. Feature Engineering

Before modeling, fields within the data that could cause problems for the model were investigated.

Target fields that could create data leakage were removed from the model inputs.

Additionally, IP addresses were excluded from the model inputs since they are considered identifiers.

To better represent the structure of network traffic, the following features were engineered:

- `has_ssl`
- `has_http`
- `bytes_ratio`
- `pkts_ratio`

Zero-variance features were also checked, and columns carrying no information were removed.

The objective of this stage is not just to reduce the column count but to produce features that represent network behavior more meaningfully.

---

## 3. Exploratory Data Analysis

Target distributions were examined to check class balance.

The distributions and outlier structures of numeric variables were analyzed separately.

It was observed that some variables in the network data contained a high volume of zero values, and some variables had a right-skewed distribution. For this reason, outlier values were not automatically deleted.

Since accepting high byte/packet values directly as data errors might not be correct—especially in network intrusion problems—RobustScaler was preferred.

The relationships between numeric variables were also examined using correlation analysis.

---

## 4. Binary Classification

The first modeling problem was framed as:

> **Is this network traffic normal or an attack?**

At this stage, instead of passing the attack type directly to the model, the attack information was handled through a binary target.

The following models were used for comparison:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Model performance was evaluated using accuracy, alongside precision, recall, and F1-score.

The F1-score was specifically taken into account, as using only accuracy can be misleading in cases where the class distribution might be imbalanced.

---

## 5. Attack Classification

After binary classification, the second problem was addressed:

> **If the traffic is an attack, which attack type does it belong to?**

At this stage, `type` was used as the target variable.

The attack types were modeled as:

- backdoor
- ddos
- dos
- injection
- mitm
- password
- ransomware
- scanning
- xss

Binary classification and attack classification were separated from each other. This allowed the intrusion detection problem and the determination of the attack type to be evaluated as distinct problems.

---

## 6. Network Traffic Analysis

The relationship between attack types and network protocols was analyzed separately.

For example, `proto` distributions were normalized by attack types to analyze which attack classes were concentrated on which protocols.

Additionally, `src_port` and `dst_port` values were examined to understand network endpoint behavior.

These variables were not interpreted directly like classic continuous variables; their meanings within the network context were taken into account.

User-Agent information was also evaluated to investigate client behaviors that might be related to attack types.

---

## 7. Preprocessing

Numeric and categorical variables were processed through separate preprocessing workflows.

For numeric variables:

- Median imputation
- RobustScaler

were used.

For categorical variables:

- Most-frequent imputation
- OneHotEncoder

were used.

By using `handle_unknown="ignore"`, errors during the test/validation stage due to unseen categories during training were prevented.

Preprocessing and the model were kept within the same Pipeline.

This structure was used to prevent preprocessing steps from creating data leakage into the validation/test data.

---

## 8. Feature Selection

Feature selection was integrated into the model pipeline.

Using `SelectFromModel` and Random Forest feature importance values, less important features were removed from the model.

Since feature selection is performed after preprocessing, features resulting from categorical encoding can also be included in the selection process.

This approach ensures a model-based selection rather than manually hardcoding a feature list.

---

## 9. Model Comparison

Models were compared under the same preprocessing and validation framework.

In the validation results, although the performances of the models for the binary problem were close to each other, tree-based models produced higher F1-scores.

Specifically, the obtained validation results showed:

**Binary Classification (Normal vs. Attack)**
- Random Forest: **F1 ≈ 0.9989**
- Decision Tree: **F1 ≈ 0.9982**
- XGBoost: **F1 ≈ 0.9951**
- Logistic Regression: **F1 ≈ 0.6448**

**Attack-Type Classification (Multiclass)**
- Random Forest: **F1 ≈ 0.9854**
- Decision Tree: **F1 ≈ 0.9811**
- XGBoost: **F1 ≈ 0.9749**
- Logistic Regression: **F1 ≈ 0.2197**

This result indicates that non-linear relationships in network traffic on this data are better captured by tree-based models compared to linear Logistic Regression.

A convergence problem was also observed for Logistic Regression. Since the convergence issue persisted despite increasing `max_iter`, the result of this model should be interpreted with extra caution.

---

## 10. Hyperparameter Tuning

Following the model comparison, hyperparameter tuning was applied to the strongest models.

Cross-validation was performed on the training data using `GridSearchCV`.

Validation/test data was not included in the tuning process.

This prevented model selection and hyperparameter optimization from being driven by the test set.

---

## 11. Model Evaluation

Final model performance was evaluated on the test set.

Using a confusion matrix, the classes that the model predicted correctly or incorrectly were examined.

Instead of looking at a single performance metric, the following were evaluated together:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Since false negative results are critical in intrusion detection problems, recall was given special consideration.

---

## 12. Model Interpretability

Feature importance analysis was conducted to understand which features the model utilized.

Feature importance was used for tree-based models, while coefficient magnitude was used for Logistic Regression.

Additionally, it was aimed to investigate how model predictions are affected by features using SHAP.

This analysis helps to understand not only whether the model makes correct predictions, but **why it makes those predictions**.

---

## 13. Key Questions

The following questions were addressed throughout the project:

- How much of the network traffic is an attack?
- What is the distribution among attack classes?
- Which network features are related to attack behavior?
- Which protocols do attack types concentrate on?
- Are there distinct differences in source/destination port behaviors?
- Does HTTP and SSL usage change depending on attack types?
- Which model is more successful for binary attack detection?
- Which model is more successful for attack type classification?
- Which features does the model find more important?
- Which attack classes are confused with others?
- On which attack types are the model's errors concentrated?
- What are the main limitations of the models developed?

---

## 14. Conclusion

In this study, rather than merely building a classification model on the TON-IoT network traffic, an end-to-end machine learning workflow was established in the order of data analysis → feature engineering → preprocessing → feature selection → model comparison → hyperparameter tuning → test evaluation → model interpretation.

The most critical approach was focusing on understanding the underlying behavior of the network data itself, instead of blindly optimizing model performance alone.

In particular, the removal of identifier fields, strict data leakage controls, network-specific feature engineering, the implementation of RobustScaler, and the strategic separation of the binary and multi-class attack classification problems formed the foundational decisions of this modeling architecture.

---

## How to Run

1. Clone the repository and navigate into it
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (macOS/Linux)
4. Install dependencies: `pip install -r requirements.txt`
5. Launch Jupyter: `jupyter notebook`
6. Open `notebook/network-ids-ton-iot-analysis.ipynb` and run all cells
   (Kernel → Restart & Run All)