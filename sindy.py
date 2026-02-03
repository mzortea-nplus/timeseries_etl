import duckdb 
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import pickle
from sindy_training import *


def predict_ts(df, model):
    df = df.copy()

    df["time"] = pd.to_datetime(df["time"])
    df["dt"] = (
        (df["time"] - df["time"].iloc[0]).dt.total_seconds() / 86400.0
    )

    tmp_sensors = [c for c in df.columns if c.endswith("_t")]
    df["tmp"] = df[tmp_sensors].mean(axis=1)

    df = df.iloc[::10].reset_index(drop=True)

    residuals = pd.DataFrame(index=df.index)
    residuals["time"] = df["time"]

    t = df["dt"].to_numpy()
    tmp = df["tmp"].to_numpy()

    for s in df.columns:
        if s in ("time", "dt", "tmp"):
            continue

        y = df[s].to_numpy()
        preds = np.zeros_like(y)
        preds[0] = y[0]

        for i in range(1, len(y)):
            X = pd.DataFrame({
                "time": [t[i-1]],
                "values": [y[i-1]],
                "tmp": [tmp[i-1]],
            })
            preds[i] = model.predict(X)[0]

        residuals[s] = y - preds

    return residuals


if __name__ == "__main__":
    conn = duckdb.connect(database='dbt/dev.duckdb', read_only=True)
    df = conn.execute("SELECT * FROM main_staging.all_static").df()
    with open('dbt/model.pkl','rb') as f:
        model = pickle.load(f)
    residuals = predict_ts(df, model)
    residuals.to_parquet('dbt/residuals.parquet')