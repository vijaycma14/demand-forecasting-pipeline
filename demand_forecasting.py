import pandas as pd


Demand_data=pd.read_csv("D:/vijay/Data scientist/Projects/Demand forecating/demand-forecasting-kernels-only/train.csv")
    #print(Demand_data.shape)= 913000 rows x 4 columns
    #print(Demand_data.head(5))
    #print(Demand_data.isnull().sum())=there is no null values 
    #print(Demand_data.dtypes)Date data type only need to chage
Demand_data["date"]= pd.to_datetime(Demand_data["date"])
    #print(Demand_data.dtypes)changed to datatime

    #Step 2 - EDA questions
    # Start and end date of the data sets
    #print("Start_date:",Demand_data['date'].min())
    #print("End_date:",Demand_data['date'].max())

    #print(Demand_data['store'].nunique(),Demand_data['item'].nunique()) To verify the data sanity check to prove you have a balanced panel dataset.
    #print(Demand_data['sales'].min())
    #print(Demand_data['sales'].max())
    #print(Demand_data['sales'].mean())

Demand_data["day"]=Demand_data["date"].dt.day
Demand_data["month"]=Demand_data["date"].dt.month
Demand_data["year"]=Demand_data["date"].dt.year
Demand_data["dayofweek"] = Demand_data["date"].dt.dayofweek
Demand_data["is_weekend"] = Demand_data["dayofweek"].apply(lambda x: 1 if x >=5 else 0)


Demand_data['sales_lag_1'] = Demand_data.groupby(['store', 'item'])['sales'].shift(1)
Demand_data['sales_lag_7'] = Demand_data.groupby(['store', 'item'])['sales'].shift(7)
Demand_data['sales_rolling_avg_7'] = Demand_data.groupby(['store', 'item'])['sales'].transform(lambda x: x.rolling(window=7).mean())
Demand_data['sales_rolling_avg_30'] = Demand_data.groupby(['store', 'item'])['sales'].transform(lambda x: x.rolling(window=30).mean())
    #print(Demand_data.head(30).tail())
Demand_data=Demand_data.dropna()
    #print(Demand_data)
#We are using chronological login here to do the Train_test_split, we purposefully avoiding the Sklear Train_test_spli here due to the shuffel and random will spoil the time series data here.
train = Demand_data[Demand_data['date'] <'2017-01-01']
test = Demand_data[Demand_data['date'] >='2017-01-01']
#print("Train shape:", train.shape)
#print("Test shape:", test.shape)

features = ['store', 'item', 'year', 'month', 'day', 'dayofweek', 'is_weekend', 
            'sales_lag_1', 'sales_lag_7',  'sales_rolling_avg_30']
target = 'sales'

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

#print("X_train shape:", X_train.shape)
#print("y_train shape:", y_train.shape)
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np

# 1. Initialize the LightGBM Regressor
model = lgb.LGBMRegressor(random_state=42)

# 2. Train the model using features and targets
model.fit(X_train, y_train)

# 3. Predict on the unseen test set
predictions = model.predict(X_test)

# 4. Evaluate the error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Test RMSE: {rmse:.4f}")   



