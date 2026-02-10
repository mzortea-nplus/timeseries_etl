import pandas as pd 
from functions import interpolation
import numpy as np
from sklearn.base import clone
from matplotlib import pyplot as plt
import mlflow

time_factor = 60 * 60 * 24 # seconds to days

def training(df, training_pipeline, debug=False):
    df["time"] = pd.to_datetime(df["time"])
    df = interpolation(df)

    tmp_sensors = [c for c in df.columns if c.endswith("_t")]
    df["tmp"] = df[tmp_sensors].mean(axis=1)

    t = (df["time"] - df["time"].iloc[0]).dt.total_seconds() / time_factor
    t = t.values


    models = []
    for s in [c for c in df.columns if c not in ["dt", "time", "month", "tmp"] + tmp_sensors]:

        model = clone(training_pipeline)
        model.t0 = df['time'].iloc[0]

        if df[s].isnull().all():
            continue
        x = df[s].values

        #x_smoothed = savgol_filter(x.to_numpy(), window_length=3, polyorder=1)
        #dxdt = [dx / dt * time_factor for dx, dt in zip(np.diff(x_smoothed)[1:], df['time'].diff().dt.total_seconds()[1:])]


        X = pd.DataFrame({
            "time": t[:-1],
            "values": x[:-1],
            "tmp": df["tmp"].iloc[:-1].values,
        })

        y = x[1:]

        model.fit(X, y)
        #predictions = integrate(model, x_smoothed[0], df['tmp'].values, t[:-1], t[1]-t[0])
        predictions = model.predict(X)
        residuals = x[:-1] - predictions
        rmse = np.sqrt(np.mean(residuals**2))
        mlflow.log_metric(f'{s}_rmse', np.sqrt(np.nanmean(residuals**2)))


        models.append(
            {
                'sensor': s,
                'model': model,
                'rmse': rmse
            }
        )

        print(s, rmse)

        if debug is True:
            plt.plot(t[:-1], x[:-1], label='measured', color='blue', alpha=0.6)
            plt.plot(t[:-1], predictions, label='predicted', color='red', alpha=0.6)
            plt.legend()
            plt.show()



    return models
