import os
import psycopg2
import sys
import pandas as pd
import numpy as np

# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from custom_class import NewFeatureTransformer

import json
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error


def connect(params_dic):
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params_dic)
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        sys.exit(1) 
    print("Connection successful")
    return conn

def get_data(query):
    return pd.read_sql_query(query, engine)

def preprocessing(data):
    df = data.drop_duplicates('TanggalInteger').dropna()
    df = df.sort_values(by='TanggalInteger')
    df['month'] = pd.to_datetime(df['TanggalInteger'],format="%Y%m%d").dt.month
    return df

def split_train_test(data,test=10):
    train = data[:-test]
    test = data[-test:]

    return train,test

# # Connection parameters
# params = {
#     "host"      : "ugems.id",
#     "port"      : 3016,
#     "database"  : "postgres",
#     "user"      : "postgres",
#     "password"  : "P@ssw0rd9901"
# }

# engine = connect(params)

# query = """
#         SELECT "TanggalInteger","Temp_surface_sea"
#         FROM sbx_bmkg.fact_bmkg_offshore
#         """
        
# train,test = split_train_test(preprocessing(get_data(query)))

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# Split data train into dependent and independent variables
X_train = train.drop('Temp_surface_sea', axis=1)
y_train = train['Temp_surface_sea']

# train model
dtr = DecisionTreeRegressor(random_state=42)
dtr.fit(X_train, y_train)

class ModelOut (mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    def predict (self, context, model_input):
        model_input.columns= map(str.lower,model_input.columns)
        return self.model.predict_proba(model_input)[:,1]
    
mlflow_conda={'channels': ['defaults'],
     'name':'conda',
     'dependencies': [ 'python=3.9', 'pip',
     {'pip':['mlflow','scikit-learn','cloudpickle','pandas','numpy']}]}

# model validation
mse_train = mean_squared_error(y_train, dtr.predict(X_train))
r2_train = r2_score(y_train, dtr.predict(X_train))
mae_train = mean_absolute_error(y_train, dtr.predict(X_train))

# validation model to JSON
train_metadata = {
    'model' : 'DecisionTreeRegressor',
    'info': 'train model result',
    'mse': mse_train,
    'r2':r2_train,
    'mae':mae_train
    }

# # Serialize and save model
# joblib.dump(dtr)

# Split data test into dependent and independent variables
X_test = test.drop('Temp_surface_sea', axis=1)
y_test = test['Temp_surface_sea']

# Predict
dtr_predictions = dtr.predict(X_test)

# model evaluatoin
mse_test = mean_squared_error(y_test, dtr_predictions)
r2_test = r2_score(y_test, dtr_predictions)
mae_test = mean_absolute_error(y_test, dtr_predictions)

# validation model to JSON
test_metadata = {
    'model' : 'DecisionTreeRegressor',
    'info': 'test model result',
    'mse': mse_test,
    'r2':r2_test,
    'mae':mae_test
    }

with mlflow.start_run():
    #log metrics
    mlflow.log_metric("mse", mse_test)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae_test)
    # log model
    mlflow.pyfunc.log_model( artifact_path="dtr_model",
      python_model=ModelOut(model=dtr,),
      code_path=['custom_class.py'],
      conda_env=mlflow_conda)
    signature = infer_signature(y_train, y_pred)
    #print out the active run ID
    run = mlflow.active_run()
    print("Active run_id: {}".format(run.info.run_id))
