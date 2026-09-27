# Subscription Customer Analytics: Spend Prediction & Churn Classification

Two end-to-end supervised learning projects built on a shared subscription-customer dataset: a **regression** model that predicts monthly spend, and a **classification** model that predicts customer churn.

## Dataset

`data/subscription_customers_dirty.csv` — ~10,000 synthetic subscription customers (demographics, usage, plan, and billing attributes). The data was generated with intentionally embedded data-quality issues (invalid values, sentinel codes, outliers, missing data) to practice realistic, production-style data cleaning rather than working with a pre-cleaned dataset.

## Project 1 — Monthly Spend Prediction (Regression)
[regression_model.ipynb](./notebooks/regression_model.ipynb)

- **Goal:** predict a customer's monthly spend (PLN) from account and usage attributes.
- **Approach:** systematic data cleaning (outlier and sentinel-value detection, with reasoning for each decision); six candidate models (Linear, Ridge, Lasso, Elastic Net, Decision Tree, XGBoost) compared via cross-validation and tuned with `GridSearchCV` / Optuna; final model chosen by CV performance rather than complexity.
- **Result:** a regularized linear model (Lasso) generalizes best — **MAE 11.43 PLN, R² 0.86** on held-out test data, a ~65% lower error than a median-prediction baseline (MAE 32.40 PLN). Subscription plan and tenure are the strongest drivers of spend.
- **Notable finding:** a small group of customers with genuine zero spend accounts for a disproportionate share of total error; their profile is otherwise unremarkable, so this is reported as a clear model limitation rather than smoothed over.

## Project 2 — Churn Prediction (Classification)
[classification_model.ipynb](./notebooks/classification_model.ipynb)

- **Goal:** predict whether a customer will churn, prioritizing catching at-risk customers over avoiding false alarms.
- **Approach:** stratified train/test split and cross-validation to handle ~81/19 class imbalance; eight models compared (including a dummy baseline) using Recall and F2 rather than accuracy; the leading model tuned with `RandomizedSearchCV`, with the classification threshold itself optimized for F2 rather than left at the default 0.5.
- **Result:** a tuned Logistic Regression model with an optimized decision threshold reaches **Recall 0.85, F2 0.59, ROC-AUC 0.75** on held-out test data, against a no-skill baseline of 0 recall. Login activity, satisfaction score, and tenure are the strongest predictors.
- **Notable finding:** the model's missed churners (false negatives) look like loyal, satisfied, long-tenured customers — a concrete blind spot called out for business stakeholders rather than hidden behind an aggregate metric.

## Docker

The project includes a lightweight Docker setup for running the saved machine-learning models in an isolated and reproducible environment. The image uses Python 3.11-slim, installs only the required runtime dependencies, and copies the prediction script together with the trained model files. When the container starts, it loads both models and runs a sample prediction for monthly spend and churn risk.

## Skills demonstrated

Data cleaning & quality assessment · EDA & visualization · `scikit-learn` pipelines (`ColumnTransformer`, no leakage between train/test) · cross-validation & model selection · hyperparameter tuning (`GridSearchCV`, `RandomizedSearchCV`, Optuna) · imbalanced classification & threshold optimization · baseline comparison · error analysis & business interpretation · Containerization (Docker) 

## Setup

```bash
pip install pandas numpy scikit-learn xgboost lightgbm optuna matplotlib seaborn
```

Place `subscription_customers_dirty.csv` in a `data/` folder, then run two notebooks from /notebooks folder
