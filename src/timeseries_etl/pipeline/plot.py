"""Plotting step: z-score plots, alert plots, summary CSV."""

import os

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from timeseries_etl.config import load_config
from timeseries_etl.domain.constants import COLORS, FONT_SIZE
from timeseries_etl.domain.events import EventsController
from timeseries_etl.domain.labels import load_label_dict


def run_plot(
    control_csv_path: str,
    year: str,
    month: str,
    opera_key: str,
    config_path: str = "configs/config_report.yaml",
) -> str:
    """Generate z-score and alert plots. Returns figure output directory."""
    config = load_config(config_path)
    site_code = config["site"]["code"]
    ym = config["data"]["month"]
    y = year or ym[:4]
    m = month or ym[5:7]
    op_key = opera_key or site_code

    fig_root = "figures"
    month_tag = f"{y}_{m}"
    fig_out = os.path.join(fig_root, op_key, month_tag)
    os.makedirs(fig_out, exist_ok=True)

    label_dict = load_label_dict(op_key)
    base_blue = COLORS["base_blue"]
    dark_blue = COLORS["dark_blue"]

    df = pd.read_csv(control_csv_path)
    t_raw = df["time"]
    if pd.api.types.is_numeric_dtype(t_raw) and t_raw.min() < 1e12:
        t = pd.date_range(
            start=f"{y}-{m}-01",
            periods=len(df),
            freq="15min",
        )[: len(df)]
    else:
        df["time"] = pd.to_datetime(df["time"])
        t = df["time"]

    summary: list[dict] = []

    for col in df.columns:
        if col == "time":
            continue

        y_arr = df[col].to_numpy()
        base_id = col.split("_")[0].upper()
        if base_id not in label_dict:
            raise ValueError(f"ID sensore '{base_id}' non presente nel file label-id")
        label = label_dict[base_id]
        suffix = col.split("_")[-1]
        axis_tag = f"_{suffix}" if suffix in ["x", "y"] else ""
        pretty_name = f"{label}{axis_tag}"

        in_range = np.abs(y_arr) <= 3
        out_range = ~in_range
        y_in = y_arr.copy()
        y_out = y_arr.copy()
        y_out[in_range] = np.nan

        plt.figure(figsize=(12, 7))
        plt.plot(t, y_in, "-", color="black", linewidth=0.8)
        plt.scatter(t, y_out, color="darkorange", s=12)
        plt.fill_between([min(t), max(t)], -3, +3, color=base_blue, alpha=0.15)
        plt.hlines([-3, +3], min(t), max(t), color=dark_blue, linestyle="--")
        plt.text(
            0.5, 0.125, "Limite warning",
            ha="center", va="center", transform=plt.gca().transAxes,
            color=dark_blue, fontsize=FONT_SIZE,
        )
        plt.text(
            0.5, 0.875, "Limite warning",
            ha="center", va="center", transform=plt.gca().transAxes,
            color=dark_blue, fontsize=FONT_SIZE,
        )
        ax = plt.gca()
        ax.set_xlabel("Tempo [gg]", fontsize=FONT_SIZE)
        ax.set_ylabel("z-score", fontsize=FONT_SIZE)
        ax.tick_params(axis="both", labelrotation=0, labelsize=FONT_SIZE)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
        ax.set_yticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
        ax.set_ylim([-5, +5])
        ax.set_title(pretty_name, fontsize=FONT_SIZE)
        ax.grid(True, which="major", axis="both", linestyle="--")
        plt.tight_layout()
        output_png = os.path.join(fig_out, f"z-score_{col}.png")
        plt.savefig(output_png, dpi=300)
        plt.close()
        print(f"\033[92m✔ salvato {output_png}\033[0m")

        n_warning = int(np.sum(np.abs(y_arr) > 3))

        controller = EventsController(z=y_arr, time=t, k=3, decay_rate=0.05, alert_th=3.0)
        alarm_series_arr, _ = controller.run()

        plt.figure(figsize=(12, 8))
        plt.plot(t, alarm_series_arr, color="black", linewidth=1.5)
        plt.axhline(3.0, color=dark_blue, linestyle="--", linewidth=1.2)
        plt.fill_between(t, 0, 3.0, color=dark_blue, alpha=0.2)
        ax = plt.gca()
        ax.set_xlabel("Tempo [gg]", fontsize=FONT_SIZE)
        ax.set_ylabel("Livello di allerta", fontsize=FONT_SIZE)
        ax.set_title(f"Livello di allerta {pretty_name}", fontsize=FONT_SIZE)
        ax.tick_params(axis="x", labelrotation=0, labelsize=FONT_SIZE)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
        ax.grid(True, linestyle="--")
        plt.tight_layout()
        output_alert_png = os.path.join(fig_out, f"alert_{col}.png")
        plt.savefig(output_alert_png, dpi=300)
        plt.close()
        print(f"\033[92m✔ salvato {output_alert_png}\033[0m")

        n_alarm = int(np.sum(alarm_series_arr >= 3.0))
        summary.append({
            "sensor_id": col,
            "label": pretty_name,
            "warnings": n_warning,
            "alarms": n_alarm,
        })

    summary_df = pd.DataFrame(summary)
    out_csv = os.path.join(fig_out, f"{y}_{m}_summary.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"\033[92m\n✔ salvato {out_csv}\033[0m")
    return fig_out
