

import pandas as pd

def load_and_preprocess_data(file_path):
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
    return Demand_data

if __name__ == "__main__":
    file_path = "D:/vijay/Data scientist/Projects/Demand forecating/demand-forecasting-kernels-only/train.csv"
    processed_df = load_and_preprocess_data(file_path)
    
    processed_df.to_csv("processed_demand_data.csv", index=False)
    print("Preprocessing complete. Saved to processed_demand_data.csv")

