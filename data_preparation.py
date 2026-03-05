import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sys
import os
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

import duckdb
import yaml

OPERE = {
    "P001": "P001_Sommacampagna",
    "P002": "P002_Giuliari_Milani",
    "P003": "P003_Gua",
    "P004": "P004_Adige_Est",
    "P005": "P005_Adige_Ovest",
}

FONT_SIZE = 20
DEG_TO_MRAD = np.pi / 180 * 1000

def get_ylabel(sensor_id: str) -> str:
    """Return pretty y-label for raw units."""
    suffix = sensor_id.split("_")[-1]
    mapping = {
        "t": "Temperatura [°C]",
        "e": "Estensione [mm]",
        "s": "Spostamento [mm]",
        "x": "Rotazione longitudinale [mrad]",
        "y": "Rotazione trasversale [mrad]",
    }
    return mapping.get(suffix, sensor_id)


def load_config(config_path: str = "configs/config_report.yaml") -> dict:
    """Load report configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_label_dict(opera_key: str) -> dict:
    """
    Load mapping ID → human-readable label from
    data/label-id/{OPERA_KEY}_label-id.csv (same logic as original script).
    """
    label_path = os.path.join("data", "label-id", f"{opera_key}_label-id.csv")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"File label-id non trovato: {label_path}")

    label_df = pd.read_csv(label_path, sep=None, engine="python")
    if label_df.shape[1] < 2:
        raise ValueError(
            f"Il file {label_path} deve avere almeno 2 colonne (label, id)"
        )

    label_df.iloc[:, 0] = label_df.iloc[:, 0].astype(str).str.strip()
    label_df.iloc[:, 1] = label_df.iloc[:, 1].astype(str).str.strip()

    label_dict = {
        row_id.split("_")[0].upper(): label
        for label, row_id in zip(label_df.iloc[:, 0], label_df.iloc[:, 1])
    }
    print(f"✔ Caricate {len(label_dict)} associazioni ID → Label")
    return label_dict


def build_data_paths(config: dict) -> tuple[str, str]:
    """Build S3 parquet glob paths from config (str_path, tmp_path, month)."""
    data_cfg = config["data"]
    site_code = config["site"]["code"]
    ym = data_cfg["month"]  # e.g. "2026-02"

    incl_pattern = None
    spost_pattern = None
    str_pattern = None
    tmp_pattern = None

    if 'incl_path' in data_cfg:
        incl_path = data_cfg["incl_path"].rstrip("/")
        incl_pattern = f"{incl_path}/{site_code}_{ym}*_rot.parquet"
    if 'spost_path' in data_cfg:
        spost_path = data_cfg["spost_path"].rstrip("/")
        spost_pattern = f"{spost_path}/{site_code}_{ym}*_dsp.parquet"
    if 'str_path' in data_cfg:
        str_path = data_cfg["str_path"].rstrip("/")
        str_pattern = f"{str_path}/{site_code}_{ym}*_str.parquet"
    if 'tmp_path' in data_cfg:
        tmp_path = data_cfg["tmp_path"].rstrip("/")
        tmp_pattern = f"{tmp_path}/{site_code}_{ym}*_tmp.parquet"

    return {'incl': incl_pattern, 'spost': spost_pattern, 'str': str_pattern, 'tmp': tmp_pattern}

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
    """
    )

def retrieve_raw_data(filepath: str):
    print(filepath)
    query = f"""
        SELECT
            time_bucket(INTERVAL '15m', datetime) AS time_range,
            AVG(COLUMNS(*))
        FROM read_parquet('{filepath}')
        GROUP BY time_range
        ORDER BY time_range;      
    """ if 'rot' in filepath else f"SELECT * from read_parquet('{filepath}')"
    if filepath is None:
        return None
    df = duckdb.sql(query).df()
    df.drop(columns=['datetime'], inplace=True)
    df = df.rename({'time_range': 'datetime'})
    print(df.head(5))
    return df

def plot_raw_data(df_raw, config):
    sensor_cols = [c for c in df_raw.columns if c not in ["time", "month"]]
    temp_sensors = [c for c in sensor_cols if c.endswith("_t")]
    struct_sensors = [c for c in sensor_cols if not c.endswith("_t")]

    opera_key = config['site']['code']
    ym = config['data']['month']
    year = ym[:4]
    month = ym[5:7]

    label_dict = load_label_dict(opera_key)

    opera_dir = OPERE.get(opera_key, f"{opera_key}_Unknown")
    fig_root = "figures"
    month_tag = f"{year}_{month}"
    fig_out = os.path.join(fig_root, opera_dir, month_tag)
    os.makedirs(fig_out, exist_ok=True)


    t_raw = df_raw['time_range'] if 'time_range' in df_raw else (df_raw['time'] if 'time' in df_raw.columns else None)
    if t_raw is None:
        raise Exception('Time column not found in raw_data')

    if len(temp_sensors) > 1:
        print("!!! Warning: the code supports only one thermometer but {len(temp_sensor)} were found. Keeping only first sensor !!!")
        temperature = np.mean(df_raw[temp_sensors], axis=1)
    else:
        temperature = df_raw[temp_sensors[0]]
    temp_col = temp_sensors[0]


    for sensor_id in struct_sensors:
        base_id = sensor_id.split("_")[0].upper()
        if base_id not in label_dict:
            continue

        label = label_dict[base_id]
        y = df_raw[sensor_id].copy()

        suffix = sensor_id.split("_")[-1]
        axis_tag = f"_{suffix}" if suffix in ["x", "y"] else ""
        if suffix in ["x", "y"]:
            y = y * DEG_TO_MRAD

        fig, ax1 = plt.subplots(figsize=(12, 5))

        ax1.plot(t_raw, y, color='blue', linewidth=1)
        ax1.set_xlabel(f"Time [gg]", fontsize=FONT_SIZE)
        ax1.set_ylabel(get_ylabel(sensor_id), color='blue', fontsize=FONT_SIZE)
        ax1.tick_params(axis='y', labelcolor='blue', labelsize=FONT_SIZE)
        ax1.xaxis.set_major_locator(
            mdates.DayLocator(bymonthday=range(5, 32, 5))
        )
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
        ax1.grid(True, linestyle="--")

        ax2 = ax1.twinx()
        ax2.plot(t_raw, temperature, color="darkorange", linewidth=1.5)
        ax2.set_ylabel(
            "Temperatura [°C]", color="darkorange", fontsize=FONT_SIZE
        )
        ax2.tick_params(
            axis="y", labelcolor="darkorange", labelsize=FONT_SIZE
        )

        temp_label = label_dict.get(
            temp_col.split("_")[0].upper(), temp_col
        )
        plt.title(f"{label}{axis_tag} e {temp_label}", fontsize=FONT_SIZE)
        plt.tight_layout()

        output_png = os.path.join(fig_out, f"raw_{label}{axis_tag}.png")
        plt.savefig(output_png, dpi=300)
        plt.close()
        print(f"\033[92m✔ salvato {output_png}\033[0m")

def thermal_model(df: pd.DataFrame):

    nan_value = float("NaN")
    df.replace("", nan_value, inplace=True)
    df.dropna(how='all', axis=1, inplace=True)
    df = df.interpolate(method='linear', axis=0)
    
    temp_sensors = [c for c in df.columns if c.endswith('_t')]
    datetime_cols = {'time', 'datetime'} | {
        c for c in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[c])
    }
    sensors = [
        c for c in df.columns
        if c not in temp_sensors
        and c not in datetime_cols
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(temp_sensors) == 0:
        print("!!! Warning: no temperature sensors found. Skipping thermal compensation !!!")
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
    label_dict = load_label_dict(config['site']['code'])

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
        time_col = "datetime" if "datetime" in d.columns else ("time" if "time" in d.columns else None)
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

    dfs = [d for d in [df_incl, df_spost, df_str, df_tmp] if d is not None and not d.empty]
    if not dfs:
        raise ValueError(f"No data found for {config['site']['code']} in {config['data']['month']}")
    df_joined = pd.concat(dfs, axis=1, join="inner")
    df_joined = df_joined[~df_joined.index.duplicated(keep="first")]
    df_joined.sort_index(inplace=True)
    # count missing timestamps
    delta_t_arr = df_joined['time_range'].diff().dt.total_seconds()
    delta_t = delta_t_arr.mode()[0]
    missing_timestamps = np.sum(np.where(delta_t_arr > 1.1 * delta_t))
    print("missing_timestamps:", missing_timestamps)
    
    # Compute NaN percentage per column
    nans_percentage = (
        df_joined.isna()
        .mean()
        .mul(100)
        .rename("dati mancanti")      # rename column
        .to_frame()
    )

    nans_percentage["label"] = (
        nans_percentage.index
        .str.split("_").str[0].str.upper()
        .map(label_dict)
    )
    # Rename index -> column
    nans_percentage = (
        nans_percentage
        .rename_axis("sensore")        # rename index
        .reset_index()
    )

    # Save
    os.makedirs("outputs", exist_ok=True)
    nans_percentage.to_csv("outputs/nans_percentage.csv", index=False)
    
    plot_raw_data(df_joined, config=config)

    # Persist raw joined data for downstream raw+temperature plots (alert_dynamics),
    # using a temporary CSV that will be deleted after use.
    os.makedirs("data", exist_ok=True)
    raw_path = f"data/raw_{site_code}_{year}_{month}.csv"
    #df_joined.to_csv(raw_path, index_label="time")

    residuals_df = thermal_model(df_joined)
    control_df = control(residuals_df)
    control_df["time"] = df_joined.index
    control_df = control_df[["time"] + [c for c in control_df.columns if c != "time"]]

    if output_path is None:
        os.makedirs("data", exist_ok=True)
        output_path = f"data/control_{site_code}_{year}_{month}.csv"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    control_df.to_csv(output_path, index=False)
    print(f"\033[92m✔ salvato {output_path}\033[0m")
    return output_path


if __name__ == "__main__":
    key_id = sys.argv[1] if len(sys.argv) > 1 else None
    secret = sys.argv[2] if len(sys.argv) > 2 else None
    config_path = sys.argv[3] if len(sys.argv) > 3 else "configs/config_report.yaml"
    run_preparation(config_path=config_path, key_id=key_id, secret=secret)
