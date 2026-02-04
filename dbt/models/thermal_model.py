from duckdb import df
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

DEBUG_MODE = False

def model(dbt, session):

    dbt.config(materialized='table')

    df = session.sql(f'''
        SELECT * EXCLUDE(month)
        FROM main_staging.all_static
    ''').df()

    nan_value = float("NaN")
    df.replace("", nan_value, inplace=True)
    df.dropna(how='all', axis=1, inplace=True)
    df = df.interpolate(method='linear', axis=0)

    #df.fillna(df.mean(), inplace=True)

    #print(df.isna())
    
    temp_sensors = [c for c in df.columns if c.endswith('_t')]
    sensors = [c for c in df.columns if c != 'time' and c not in temp_sensors]

    if len(temp_sensors) == 0:
        print("!!! Warning: no temperature sensors found. Skipping thermal compensation !!!")
        return df[['time'] + sensors]

    X = df[temp_sensors].to_numpy().reshape(df.shape[0], -1)
    residuals_df = pd.DataFrame({'time': df['time']})
    for s in sensors:

        y = df[s].to_numpy()

        reg = LinearRegression()
        reg.fit(X, y)

        y_pred = reg.predict(X)
        if DEBUG_MODE:
            residuals_df[s] = y
            residuals_df[s + '_predicted'] = y_pred
        else:
            residuals_df[s] = y - y_pred

    return residuals_df

        


