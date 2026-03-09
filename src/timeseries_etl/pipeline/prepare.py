"""Data preparation step: S3 fetch, thermal model, control CSV, raw plots."""

import os

import numpy as np
import pandas as pd

from timeseries_etl.config import build_data_paths, load_config
from timeseries_etl.domain import load_label_dict
from timeseries_etl.domain.constants import COLORS
from timeseries_etl.domain.prep import control, thermal_model
from timeseries_etl.io.duckdb_s3 import retrieve_raw_data, start_s3_connection
from timeseries_etl.pipeline.prep_plots import plot_raw_data


def _set_datetime_index(df: pd.DataFrame) -> None:
    """Set datetime index and sort."""
    if df is None or df.empty:
        return
    time_col = (
        "datetime"
        if "datetime" in df.columns
        else ("time" if "time" in df.columns else None)
    )
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df.set_index(time_col, inplace=True)
        df.sort_index(inplace=True)


def run_prepare(
    config_path: str = "configs/config_report.yaml",
    key_id: str | None = None,
    secret: str | None = None,
    output_path: str | None = None,
) -> str:
    """
    Run data preparation: load config, fetch from S3, thermal model, control.
    Returns path to output control CSV.
    """
    config = load_config(config_path)
    patterns = build_data_paths(config)
    site_code = config["site"]["code"]
    ym = config["data"]["month"]
    year, month = ym[:4], ym[5:7]

    label_dict = load_label_dict(site_code)

    key_id = key_id or os.environ.get("AWS_ACCESS_KEY_ID", "")
    secret = secret or os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    start_s3_connection(key_id=key_id, secret=secret)

    df_incl = df_spost = df_str = df_tmp = None

    for key, pattern in patterns.items():
        if pattern is None:
            continue
        df = retrieve_raw_data(pattern)
        if df is None or df.empty:
            continue
        _set_datetime_index(df)
        if key == "incl":
            df_incl = df
        elif key == "spost":
            df_spost = df
        elif key == "str":
            df_str = df
        elif key == "tmp":
            df_tmp = df

    dfs = [
        d
        for d in [df_incl, df_spost, df_str, df_tmp]
        if d is not None and not d.empty
    ]
    if not dfs:
        raise ValueError(
            f"No data found for {config['site']['code']} in {config['data']['month']}"
        )

    df_joined = pd.concat(dfs, axis=1, join="inner")
    df_joined = df_joined[~df_joined.index.duplicated(keep="first")]
    df_joined.sort_index(inplace=True)
    df_joined = df_joined.reset_index()
    if "index" in df_joined.columns:
        df_joined = df_joined.rename(columns={"index": "datetime"})

    delta_t_arr = df_joined["datetime"].diff().dt.total_seconds()
    delta_t = delta_t_arr.mode()[0]
    missing_timestamps = int(np.sum(np.where(delta_t_arr > 1.1 * delta_t, 1, 0)))
    print("missing_timestamps:", missing_timestamps)

    nans_percentage = (
        df_joined.isna()
        .mean()
        .mul(100)
        .rename("dati mancanti")
        .to_frame()
    )
    nans_percentage["label"] = (
        nans_percentage.index.str.split("_").str[0].str.upper().map(label_dict)
    )
    nans_percentage = nans_percentage.rename_axis("sensore").reset_index()

    os.makedirs("outputs", exist_ok=True)
    nans_percentage.to_csv("outputs/nans_percentage.csv", index=False)

    plot_raw_data(df_joined, config, COLORS, label_dict)

    residuals_df = thermal_model(df_joined)
    control_df = control(residuals_df)
    time_col = "datetime" if "datetime" in df_joined.columns else "time"
    control_df["time"] = df_joined[time_col] if time_col in df_joined.columns else df_joined.index
    control_df = control_df[["time"] + [c for c in control_df.columns if c != "time"]]

    if output_path is None:
        os.makedirs("data", exist_ok=True)
        output_path = f"data/control_{site_code}_{year}_{month}.csv"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    control_df.to_csv(output_path, index=False)
    print(f"\033[92m✔ salvato {output_path}\033[0m")
    return output_path
