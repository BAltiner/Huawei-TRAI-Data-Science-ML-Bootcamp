# Customer Churn Prediction

This project predicts whether a telecom customer will churn. It is designed as a
small, reproducible machine-learning project rather than a notebook-only demo.

## Business question

Which current customers are likely to leave, so that a retention team can
prioritize outreach? Churn is treated as the positive class (`1`), so F1 is the
selection metric: it balances catching at-risk customers (recall) with avoiding
unnecessary offers (precision).

## What the pipeline does

1. Loads the Telco dataset and standardizes blank values.
2. Converts `TotalCharges` to numeric and removes the non-generalizable `customerID`.
3. Creates transparent risk features: month-to-month contract, technical support,
   new-customer status, and average monthly spend.
4. Makes stratified (%) 60/20/20 train/validation/test splits.
5. Fits imputers, scaling, and one-hot encoding on the training split only.
6. Compares Logistic Regression, K-Nearest Neighbors, and Decision Tree.
7. Selects the best model using validation F1, then uses the test split once for
   the final unbiased estimate.

## Results

Validation set performance (imbalanced target, ~27% churn — F1 used for selection):

| Model                | Accuracy | Precision | Recall | F1     |
|-----------------------|----------|-----------|--------|--------|
| **Logistic Regression** | 0.744    | 0.512     | 0.778  | **0.618** |
| KNN                    | 0.761    | 0.552     | 0.532  | 0.542  |
| Decision Tree          | 0.734    | 0.500     | 0.519  | 0.503  |

**Best model: Logistic Regression** (highest validation F1). It trades some
precision for much higher recall — catching more true churners at the cost of
more false alarms — which fits the retention use case, where missing an at-risk
customer is costlier than an unnecessary outreach.

## Key insights (EDA)

- Overall churn rate is ~27%, an imbalanced target.
- Gender has almost no effect on churn; SeniorCitizen, no-Partner, and
  no-Dependents customers churn more.
- Highest-risk segment: month-to-month contract + Fiber optic internet +
  Electronic check payment, especially with low tenure.
- Customers without OnlineSecurity/TechSupport churn more than those with it.
- Churned customers have lower average tenure and higher average MonthlyCharges
  than retained customers.

## Run

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

The script prints validation and final-test metrics and writes a confusion matrix
to `figures/confusion_matrix.png`.


[def]: figures/confusion_matrix.png