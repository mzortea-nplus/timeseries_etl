import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import pickle
import numpy as np
import duckdb

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
    def __init__(self, config):

        scaler = StandardScaler()

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


full_pipeline = Pipeline(
    [
        (
            "features_pipeline",
            features_pipeline(
                config = {
                    'scaler': 'StandardScaler', 
                    'time_periods': [0.5, 1, 3, 365.25, 7, 14, 30, 90]
                }
            ).full_pipeline,
        ),
        ("regressor", Lasso(alpha=0.1, fit_intercept=True)),
    ]
)

def training(df):
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
    conn = duckdb.connect(database='dbt/dev.duckdb', read_only=True)
    df = conn.execute("SELECT * FROM main_staging.all_static").df()
    residuals, full_pipeline = training(df)
    with open('dbt/model.pkl','wb') as f:
        pickle.dump(full_pipeline, f)