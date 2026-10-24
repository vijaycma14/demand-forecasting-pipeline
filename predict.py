import pandas as pd
import numpy as np
import joblib

# Load the saved model
print("Loading trained model...")
model = joblib.load("demand_model.pkl")

# Load data for prediction (using test portion of processed data)
print("Loading data for inference...")
Demand_data = pd.read_csv("processed_demand_data.csv")
Demand_data["date"] = pd.to_datetime(Demand_data["date"])
test = Demand_data[Demand_data['date'] >= '2017-01-01']

features = ['store', 'item', 'year', 'month', 'day', 'dayofweek', 'is_weekend', 
            'sales_lag_1', 'sales_lag_7', 'sales_rolling_avg_30']

X_test = test[features]

# Generate predictions
print("Generating predictions...")
predictions = model.predict(X_test)

# Display a sample of actual vs predicted sales
results = test[['date', 'store', 'item', 'sales']].copy()
results['predicted_sales'] = predictions
print("\nSample Predictions:")
print(results.head(10))