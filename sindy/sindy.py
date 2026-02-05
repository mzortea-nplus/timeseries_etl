import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import pickle
import numpy as np
import sys
import duckdb
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
import yaml
import os

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
    measured = pd.DataFrame(index=df.index)
    prediction = pd.DataFrame(index=df.index)
    residuals["time"] = df["time"]
    measured['time'] = df['time']
    prediction['time'] = df['time']

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

        measured[s] = y
        prediction[s] = preds
        residuals[s] = y - preds

    return residuals, measured, prediction

def interpolation(x, method="linear"):
    if len(x) >= 3:
        return x.bfill().ffill().interpolate(method=method, limit_direction="both")
    else:
        return x
    
def fourier_features(x, periods):
    t = np.asarray(x).flatten()
    feats = []
    for p in periods:
        w = 2 * np.pi / p
        feats.append(np.sin(w * t))
        feats.append(np.cos(w * t))
    return np.vstack(feats).T


class features_pipeline:
    def __init__(self, scaler_type, time_periods):

        if scaler_type == 'Standard':
            scaler = StandardScaler()
        else:
            scaler = None

        self.state_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean", fill_value=0)),
            ("scaler", scaler),
            ("features", PolynomialFeatures(degree=1, include_bias=False)),
        ])

        self.temperature_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean", fill_value=0)),
            ("scaler", scaler),
            ("features", PolynomialFeatures(degree=1, include_bias=False)),
        ])

        self.time_pipeline = Pipeline(
            [
                ("fourier",
                 FunctionTransformer(
                     fourier_features,
                     kw_args={"periods": config.get("time_periods", [24, 168])},
                 )),
            ]
        )

        self.full_pipeline = ColumnTransformer(
            [
                ("state", self.state_pipeline, ["values"]),
                ("control", self.temperature_pipeline, ["tmp"]),
                ("time", self.time_pipeline, ["time"]),
            ]
        )

## TEST

def training(df, full_pipeline):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])

    df = df[df["time"] < "2025-11-01"]

    tmp_sensors = [c for c in df.columns if c.endswith("_t")]
    df["tmp"] = df[tmp_sensors].mean(axis=1)

    df["dt"] = (
        (df["time"] - df["time"].iloc[0]).dt.total_seconds() / 86400.0
    )

    residuals_df = pd.DataFrame()

    for s in [c for c in df.columns if c not in ["dt", "time", "month", "tmp"] + tmp_sensors]:

        if df[s].isnull().all():
            continue
        series = interpolation(df[s])

        X = pd.DataFrame({
            "time": df["dt"].iloc[:-1].values,
            "values": series.iloc[:-1].values,
            "tmp": df["tmp"].iloc[:-1].values,
        })

        y = series.iloc[1:].values

        full_pipeline.fit(X, y)

        residuals_df[s] = y - full_pipeline.predict(X)

    residuals_df["time"] = df["time"].iloc[1:].values
    return residuals_df, full_pipeline


if __name__ == "__main__":
    with open(sys.argv[1], 'r') as f:
        config = yaml.safe_load(f)
    conn = duckdb.connect(database=config['data']['db'], read_only=True)
    df = conn.execute("SELECT * FROM main_staging.all_static").df()

    if config['mode'] == 'training':
        full_pipeline = Pipeline(
            [
                (
                    "features_pipeline",
                    features_pipeline(
                        scaler_type = None, 
                        time_periods = config['sindy']['fourier']
                    ).full_pipeline
                ),
                (
                    "regressor", 
                    Lasso(alpha=0.1, fit_intercept=True)
                ),
            ]
        )  
        residuals, full_fitted_pipeline = training(df, full_pipeline=full_pipeline)
        with open(config['out']['model'],'wb') as f:
            pickle.dump(full_pipeline, f)
    elif config['mode'] == 'prediction':
        with open(config['out']['model'],'rb') as f:
            model = pickle.load(f)
        residuals, measured, predicted = predict_ts(df, model)
        os.makedirs('figures/sindy', exist_ok=True)
        for s in residuals.columns:
            if s != 'time':
                plt.plot(measured['time'], measured[s], color='blue', label='measured')
                plt.plot(predicted['time'], predicted[s], color='red', label='predicted')
                plt.legend()
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%a %d'))

                #plt.xlim([min(t) - timedelta(days=10), max(t) + timedelta(days=10)])
                plt.xticks(rotation=30)
                plt.grid(linestyle='--')
                plt.tight_layout()
                plt.savefig(f'figures/sindy/sensor_{s}.png')
                plt.close()
                plt.show()

        residuals.to_parquet(config['out']['residuals'])
    else:
        raise Exception("In file sindy.yaml: mode should be 'training' or 'prediction'.") 