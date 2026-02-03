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

    print(df.head())
    
    temp_sensors = [c for c in df.columns if c.endswith('_t')]
    sensors = [c for c in df.columns if c != 'time' and c not in temp_sensors]

    df = df.interpolate(method='linear', axis=0)
    X = df[temp_sensors].to_numpy()
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

        


