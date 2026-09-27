import pandas as pd
import joblib
regression_model = joblib.load("models/regression_model.pkl")

classification_bundle = joblib.load("models/classification_model.pkl")
classification_model = classification_bundle["model"]
classification_threshold = classification_bundle["threshold"]

example_customer = pd.DataFrame([{
    "age": 34,
    "city": "warszawa",
    "monthly_income_pln": 8500.0,
    "tenure_months": 18,
    "avg_logins_last_30d": 9,
    "support_tickets_last_30d": 1,
    "satisfaction_score": 7,
    "plan": "standard",
    "auto_renew": 1,
    "discount_pct": 0,
    "acquisition_channel": "organic",
}])

predicted_spend = regression_model.predict(example_customer)[0]
print(f"Predicted monthly spend: {predicted_spend:.2f} PLN")

churn_probability = classification_model.predict_proba(example_customer)[:, 1][0]
will_churn = churn_probability >= classification_threshold
print(f"Churn probability: {churn_probability:.2%} (decision threshold: {classification_threshold:.2f})")
print(f"Predicted churn: {'YES' if will_churn else 'no'}")
