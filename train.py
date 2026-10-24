import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib

print("Loading processed data for training...")
Demand_data = pd.read_csv("processed_demand_data.csv")
Demand_data["date"] = pd.to_datetime(Demand_data["date"])

# Chronological Split
train = Demand_data[Demand_data['date'] < '2017-01-01']
test = Demand_data[Demand_data['date'] >= '2017-01-01']

features = ['store', 'item', 'year', 'month', 'day', 'dayofweek', 'is_weekend', 
            'sales_lag_1', 'sales_lag_7', 'sales_rolling_avg_30']
target = 'sales'

X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

# Initialize and train LightGBM model
print("Training LightGBM model...")
model = lgb.LGBMRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Test RMSE: {rmse:.4f}")

# Save the trained model for production
joblib.dump(model, "demand_model.pkl")
print("Model successfully trained and saved as demand_model.pkl")