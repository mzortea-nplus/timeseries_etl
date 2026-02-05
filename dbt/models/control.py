import duckdb 
from matplotlib import pyplot as plt


conn = duckdb.connect(database=':memory:')

def z_score(series):
    mean = series.mean()
    std = series.std()
    return (series - mean) / std

def model(dbt, session):

    dbt.config(materialized='table')
    try:
        control_df = session.sql('''
            SELECT *
            FROM thermal_model
        ''').df()
    except:
        control_df = session.sql('''
            SELECT *
            FROM main_staging.after_sindy
        ''').df()
    sensors = [c for c in control_df.columns if c != 'time']

    zscore_df = control_df.copy()
    for s in sensors:
        zscore_df[s] = z_score(control_df[s])

    return zscore_df