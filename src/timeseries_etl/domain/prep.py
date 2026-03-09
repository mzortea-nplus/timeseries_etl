"""Thermal model and control (z-score) computations."""

import warnings
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression

from timeseries_etl.domain.stats import z_score


def _split_features_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return (features_dataframe, sensor_columns) from a joined dataframe."""
    temp_sensors = [c for c in df.columns if c.endswith("_t")]
    datetime_cols = {"time", "datetime"} | {
        c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])
    }
    sensors = [
        c
        for c in df.columns
        if c not in temp_sensors
        and c not in datetime_cols
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    X = df[temp_sensors].to_numpy().reshape(df.shape[0], -1) if temp_sensors else None
    return (pd.DataFrame(X, index=df.index, columns=temp_sensors) if X is not None else pd.DataFrame()), sensors


def _load_model_for_sensor(
    sensor: str,
    models: Mapping[str, RegressorMixin] | None = None,
    model_dir: str | None = None,
) -> RegressorMixin | None:
    """Return a pre-fitted sklearn regressor for the given sensor, or None if not found."""
    if models and sensor in models:
        return models[sensor]
    if model_dir:
        path = Path(model_dir) / f"{sensor}.joblib"
        if path.exists():
            return joblib.load(path)
    return None


def thermal_model(
    df: pd.DataFrame,
    models: Mapping[str, RegressorMixin] | None = None,
    model_dir: str | None = None,
) -> pd.DataFrame:
    """
    Apply thermal compensation by regressing each structural sensor on temperature.

    - Models MUST be pre-fitted (provided via `models` or `model_dir`).
    - This function never fits models internally.
    """
    nan_val = float("nan")
    df = df.replace("", nan_val).copy()
    df = df.dropna(how="all", axis=1)
    df = df.interpolate(method="linear", axis=0)

    if models is None and model_dir is None:
        model_dir = "models"

    X, sensors = _split_features_targets(df)
    if X.empty or not sensors:
        return df[sensors].copy() if sensors else pd.DataFrame(index=df.index)

    residuals_df = pd.DataFrame(index=df.index)
    for s in sensors:
        y = df[s].to_numpy()
        model = _load_model_for_sensor(s, models=models, model_dir=model_dir)
        if model is None:
            warnings.warn(f"No pre-fitted model for sensor '{s}'; falling back to linear fit on temperature.", UserWarning)
            model = LinearRegression()
            model.fit(X, y)
        y_pred = model.predict(X)
        residuals_df[s] = y - y_pred

    return residuals_df


def thermal_model_with_predictions(
    df: pd.DataFrame,
    models: Mapping[str, RegressorMixin] | None = None,
    model_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute predictions and residuals for each sensor using arbitrary sklearn models.

    Returns (predictions_df, residuals_df, sigma_df) where:
    - predictions_df: model predictions
    - residuals_df: y - y_pred
    - sigma_df: per-sensor residual standard deviation (for error bars)
    """
    nan_val = float("nan")
    df = df.replace("", nan_val).copy()
    df = df.dropna(how="all", axis=1)
    df = df.interpolate(method="linear", axis=0)

    if models is None and model_dir is None:
        model_dir = "models"

    X, sensors = _split_features_targets(df)
    if X.empty or not sensors:
        empty = pd.DataFrame(index=df.index)
        return empty, empty, empty

    preds_df = pd.DataFrame(index=df.index)
    residuals_df = pd.DataFrame(index=df.index)
    sigma_df = pd.DataFrame(index=["sigma"])

    for s in sensors:
        y = df[s].to_numpy()
        model = _load_model_for_sensor(s, models=models, model_dir=model_dir)
        if model is None:
            warnings.warn(f"No pre-fitted model for sensor '{s}'; falling back to linear fit on temperature.", UserWarning)
            model = LinearRegression()
            model.fit(X, y)
        y_pred = model.predict(X)
        preds_df[s] = y_pred
        residuals = y - y_pred
        residuals_df[s] = residuals
        sigma_df.at["sigma", s] = float(np.std(residuals, ddof=1))

    return preds_df, residuals_df, sigma_df


def control(residuals_df: pd.DataFrame) -> pd.DataFrame:
    """Compute z-scores for each sensor column."""
    zscore_df = pd.DataFrame()
    for s in residuals_df.columns:
        zscore_df[s] = z_score(residuals_df[s].values)
    return zscore_df
