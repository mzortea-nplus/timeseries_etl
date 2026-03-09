"""Raw data plots produced during data preparation."""

import os

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from timeseries_etl.domain.constants import COLORS, DEG_TO_MRAD, FONT_SIZE
from timeseries_etl.domain.labels import get_ylabel


def plot_raw_data(
    df_raw: pd.DataFrame,
    config: dict,
    params: dict,
    label_dict: dict[str, str],
) -> None:
    """Plot raw sensor vs temperature for each structural sensor."""
    sensor_cols = [c for c in df_raw.columns if c not in ["time", "month", "datetime"]]
    temp_sensors = [c for c in sensor_cols if c.endswith("_t")]
    struct_sensors = [c for c in sensor_cols if not c.endswith("_t")]

    opera_key = config["site"]["code"]
    ym = config["data"]["month"]
    year, month = ym[:4], ym[5:7]
    fig_root = "figures"
    month_tag = f"{year}_{month}"
    fig_out = os.path.join(fig_root, opera_key, month_tag)
    os.makedirs(fig_out, exist_ok=True)

    t_raw = (
        df_raw["time_range"]
        if "time_range" in df_raw.columns
        else (df_raw["datetime"] if "datetime" in df_raw.columns else None)
    )
    if t_raw is None:
        t_raw = df_raw.index if hasattr(df_raw.index, "to_numpy") else None
    if t_raw is None:
        raise ValueError("Time column not found in raw_data")

    if len(temp_sensors) > 1:
        temperature = np.mean(df_raw[temp_sensors], axis=1)
    elif temp_sensors:
        temperature = df_raw[temp_sensors[0]]
    else:
        return
    temp_col = temp_sensors[0]

    base_blue = params.get("base_blue", COLORS["base_blue"])

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
        ax1.plot(t_raw, y, color=base_blue, linewidth=1)
        ax1.set_xlabel("Tempo [gg]", fontsize=FONT_SIZE)
        ax1.set_ylabel(get_ylabel(sensor_id), color=base_blue, fontsize=FONT_SIZE)
        ax1.tick_params(axis="y", labelcolor=base_blue, labelsize=FONT_SIZE)
        ax1.xaxis.set_major_locator(mdates.DayLocator(bymonthday=range(5, 32, 5)))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
        ax1.grid(True, linestyle="--")

        ax2 = ax1.twinx()
        ax2.plot(t_raw, temperature, color="darkorange", linewidth=1.5)
        ax2.set_ylabel("Temperatura [°C]", color="darkorange", fontsize=FONT_SIZE)
        ax2.tick_params(axis="y", labelcolor="darkorange", labelsize=FONT_SIZE)

        temp_label = label_dict.get(temp_col.split("_")[0].upper(), temp_col)
        plt.title(f"{label}{axis_tag} e {temp_label}", fontsize=FONT_SIZE)
        plt.tight_layout()

        output_png = os.path.join(fig_out, f"raw_{label}{axis_tag}.png")
        plt.savefig(output_png, dpi=300)
        plt.close()
        print(f"\033[92m✔ salvato {output_png}\033[0m")
