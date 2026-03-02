import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression


def load_config(config_path: str = "configs/config_report.yaml") -> dict:
    """Load report configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_data_paths(config: dict) -> dict:
    """Build S3 parquet glob paths from config (str_path, tmp_path, month)."""
    data_cfg = config["data"]
    site_code = config["site"]["code"]
    ym = data_cfg["month"]  # e.g. "2026-02"

    incl_pattern = None
    spost_pattern = None
    str_pattern = None
    tmp_pattern = None

    if "incl_path" in data_cfg:
        incl_path = data_cfg["incl_path"].rstrip("/")
        incl_pattern = f"{incl_path}/{site_code}_{ym}*_rot.parquet"
    if "spost_path" in data_cfg:
        spost_path = data_cfg["spost_path"].rstrip("/")
        spost_pattern = f"{spost_path}/{site_code}_{ym}*_dsp.parquet"
    if "str_path" in data_cfg:
        str_path = data_cfg["str_path"].rstrip("/")
        str_pattern = f"{str_path}/{site_code}_{ym}*_str.parquet"
    if "tmp_path" in data_cfg:
        tmp_path = data_cfg["tmp_path"].rstrip("/")
        tmp_pattern = f"{tmp_path}/{site_code}_{ym}*_tmp.parquet"

    return {
        "incl": incl_pattern,
        "spost": spost_pattern,
        "str": str_pattern,
        "tmp": tmp_pattern,
    }


def start_s3_connection(key_id, secret):
    duckdb.sql("INSTALL httpfs")
    duckdb.sql("LOAD httpfs")
    duckdb.sql(f"""
        CREATE OR REPLACE SECRET s3_secret (
            TYPE s3,
            PROVIDER config,
            KEY_ID '{key_id}',
            SECRET '{secret}',
            REGION 'eu-central-1'
        )
    """)


def retrieve_raw_data(filepath: str):
    print(filepath)
    query = (
        f"""
        SELECT
            time_bucket(INTERVAL '15m', datetime) AS time_range,
            AVG(COLUMNS(*))
        FROM read_parquet('{filepath}')
        GROUP BY time_range
        ORDER BY time_range;
    """
        if "rot" in filepath
        else f"SELECT * from read_parquet('{filepath}')"
    )
    if filepath is None:
        return None
    df = duckdb.sql(query).df()
    df.drop(columns=["datetime"], inplace=True)
    df = df.rename({"time_range": "datetime"})
    print(df.head(5))
    return df


def thermal_model(df: pd.DataFrame):

    nan_value = float("NaN")
    df.replace("", nan_value, inplace=True)
    df.dropna(how="all", axis=1, inplace=True)
    df = df.interpolate(method="linear", axis=0)

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

    if len(temp_sensors) == 0:
        print(
            "!!! Warning: no temperature sensors found. Skipping thermal compensation !!!"
        )
        return df[sensors].copy()

    X = df[temp_sensors].to_numpy().reshape(df.shape[0], -1)
    residuals_df = pd.DataFrame()
    for s in sensors:
        y = df[s].to_numpy()

        reg = LinearRegression()
        reg.fit(X, y)

        y_pred = reg.predict(X)
        residuals_df[s] = y - y_pred

    return residuals_df


def z_score(series):
    mean = series.mean()
    std = series.std()
    return (series - mean) / std


def control(residuals_df: pd.DataFrame):
    sensors = residuals_df.columns
    zscore_df = pd.DataFrame()
    for s in sensors:
        zscore_df[s] = z_score(residuals_df[s].values)

    return zscore_df


def run_preparation(
    config_path: str = "configs/config_report.yaml",
    key_id: str | None = None,
    secret: str | None = None,
    output_path: str | None = None,
) -> str:
    """
    Run data preparation: load config, fetch data from S3, run thermal model and control.
    Returns the path to the output control CSV.
    """
    config = load_config(config_path)
    patterns = build_data_paths(config)
    site_code = config["site"]["code"]
    ym = config["data"]["month"]
    year, month = ym[:4], ym[5:7]

    if key_id and secret:
        start_s3_connection(key_id=key_id, secret=secret)
    else:
        # Assume AWS credentials are set via env (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
        start_s3_connection(
            key_id=os.environ.get("AWS_ACCESS_KEY_ID", ""),
            secret=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        )

    df_incl = None
    df_spost = None
    df_str = None
    df_tmp = None

    def _set_datetime_index(d):
        if d is None:
            return
        time_col = (
            "datetime"
            if "datetime" in d.columns
            else ("time" if "time" in d.columns else None)
        )
        if time_col:
            d[time_col] = pd.to_datetime(d[time_col])
            d.set_index(time_col, inplace=True)
            d.sort_index(inplace=True)

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
        d for d in [df_incl, df_spost, df_str, df_tmp] if d is not None and not d.empty
    ]
    if not dfs:
        raise ValueError(
            f"No data found for {config['site']['code']} in {config['data']['month']}"
        )
    df_joined = pd.concat(dfs, axis=1, join="inner")
    df_joined = df_joined[~df_joined.index.duplicated(keep="first")]
    df_joined.sort_index(inplace=True)

    # Persist raw joined data for downstream raw+temperature plots (alert_dynamics),
    # using a temporary CSV that will be deleted after use.
    os.makedirs("data", exist_ok=True)
    raw_path = f"data/raw_{site_code}_{year}_{month}.csv"
    df_joined.to_csv(raw_path, index_label="time")

    residuals_df = thermal_model(df_joined)
    control_df = control(residuals_df)
    control_df["time"] = df_joined.index
    control_df = control_df[["time"] + [c for c in control_df.columns if c != "time"]]

    if output_path is None:
        os.makedirs("data", exist_ok=True)
        output_path = f"data/control_{site_code}_{year}_{month}.csv"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    control_df.to_csv(output_path, index=False)
    df_joined.to_csv(output_path.replace("control", ""), index=False)
    print(f"\033[92m✔ salvato {output_path}\033[0m")
    return output_path


if __name__ == "__main__":
    key_id = sys.argv[1] if len(sys.argv) > 1 else None
    secret = sys.argv[2] if len(sys.argv) > 2 else None
    config_path = sys.argv[3] if len(sys.argv) > 3 else "configs/config_report.yaml"
    run_preparation(config_path=config_path, key_id=key_id, secret=secret)
