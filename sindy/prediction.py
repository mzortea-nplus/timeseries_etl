import pandas as pd 
import numpy as np 

time_factor = 60 * 60 * 24

def predict_ts(df, model):
    df = df.copy()

    df["time"] = pd.to_datetime(df["time"])
    df["dt"] = (
        (df["time"] - model.t0).dt.total_seconds() / time_factor
    )

    tmp_sensors = [c for c in df.columns if c.endswith("_t")]
    df["tmp"] = df[tmp_sensors].mean(axis=1)

    t = df["dt"].to_numpy()
    tmp = df["tmp"].to_numpy()
    x = df['val'].to_numpy()

    preds = np.zeros_like(x)
    preds[0] = x[0]

    for i in range(1, len(x)):
        X = pd.DataFrame({
            "time": [t[i]],
            "values": [preds[i-1]],
            "tmp": [tmp[i]],
        })
        preds[i] = model.predict(X)[0]

    return preds

