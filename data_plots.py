import os
import sys

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

from data_preparation import load_label_dict, get_ylabel

with open("parameters.yaml", "r") as f:
    params = yaml.safe_load(f)

# ======================================================
# INPUT DATA / CONFIG
# ======================================================

OPERE = {
    "P001": "P001_Sommacampagna",
    "P002": "P002_Giuliari_Milani",
    "P003": "P003_Gua",
    "P004": "P004_Adige_Est",
    "P005": "P005_Adige_Ovest",
}

FONT_SIZE = 20
DEG_TO_MRAD = np.pi / 180 * 1000


def load_config(config_path: str = "configs/config_report.yaml") -> dict:
    """Load report configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)




def run_data_plots(
    control_csv_path: str | None = None,
    year: str | None = None,
    month: str | None = None,
    opera_key: str | None = None,
    config_path: str = "configs/config_report.yaml",
) -> str:
    """
    Generate alert dynamics plots from control CSV (output of data_preparation).
    If control_csv_path, year, month, or opera_key are omitted, they are read from config.
    Returns the figure output directory.
    """
    # --------------------------------------------------
    # Parse inputs from config
    # --------------------------------------------------
    config = load_config(config_path)
    site_code = config["site"]["code"]
    ym = config["data"]["month"]

    if opera_key is None:
        opera_key = site_code
    if year is None:
        year = ym[:4]
    if month is None:
        month = ym[5:7]
    if control_csv_path is None:
        control_csv_path = f"data/control_{site_code}_{year}_{month}.csv"

    opera_dir = OPERE.get(opera_key, f"{opera_key}_Unknown")
    fig_root = "figures"
    month_tag = f"{year}_{month}"
    fig_out = os.path.join(fig_root, opera_dir, month_tag)
    os.makedirs(fig_out, exist_ok=True)

    label_dict = load_label_dict(opera_key)

    # --------------------------------------------------
    # Raw-data + temperature plots from persisted CSV (if available)
    # --------------------------------------------------
    # raw_path = os.path.join("data", f"raw_{site_code}_{year}_{month}.csv")
    # if os.path.exists(raw_path):
    #     try:
    #         df_raw = pd.read_csv(raw_path)
    #         if "time" in df_raw.columns:
    #             t_raw = pd.to_datetime(df_raw["time"])
    #         else:
    #             t_raw = pd.date_range(
    #                 start=f"{year}-{month}-01", periods=len(df_raw), freq="15min"
    #             )

    #         sensor_cols = [c for c in df_raw.columns if c not in ["time", "month"]]
    #         temp_sensors = [c for c in sensor_cols if c.endswith("_t")]
    #         struct_sensors = [c for c in sensor_cols if not c.endswith("_t")]

    #         if temp_sensors:
    #             temp_col = temp_sensors[0]
    #             temperature = df_raw[temp_col]

    #             for sensor_id in struct_sensors:
    #                 base_id = sensor_id.split("_")[0].upper()
    #                 if base_id not in label_dict:
    #                     continue

    #                 label = label_dict[base_id]
    #                 y = df_raw[sensor_id].copy()

    #                 suffix = sensor_id.split("_")[-1]
    #                 axis_tag = f"_{suffix}" if suffix in ["x", "y"] else ""
    #                 if suffix in ["x", "y"]:
    #                     y = y * DEG_TO_MRAD

    #                 fig, ax1 = plt.subplots(figsize=(12, 5))

    #                 ax1.plot(t_raw, y, color='blue', linewidth=1)
    #                 ax1.set_xlabel(f"{year}_{month} [gg]", fontsize=FONT_SIZE)
    #                 ax1.set_ylabel(get_ylabel(sensor_id), color='blue', fontsize=FONT_SIZE)
    #                 ax1.tick_params(axis='y', labelcolor='blue', labelsize=FONT_SIZE)
    #                 ax1.xaxis.set_major_locator(
    #                     mdates.DayLocator(bymonthday=range(5, 32, 5))
    #                 )
    #                 ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
    #                 ax1.grid(True, linestyle="--")

    #                 ax2 = ax1.twinx()
    #                 ax2.plot(t_raw, temperature, color="darkorange", linewidth=1.5)
    #                 ax2.set_ylabel(
    #                     "Temperatura [°C]", color="darkorange", fontsize=FONT_SIZE
    #                 )
    #                 ax2.tick_params(
    #                     axis="y", labelcolor="darkorange", labelsize=FONT_SIZE
    #                 )

    #                 temp_label = label_dict.get(
    #                     temp_col.split("_")[0].upper(), temp_col
    #                 )
    #                 plt.title(f"{label}{axis_tag} e {temp_label}", fontsize=FONT_SIZE)
    #                 plt.tight_layout()

    #                 output_png = os.path.join(fig_out, f"raw_{label}{axis_tag}.png")
    #                 plt.savefig(output_png, dpi=300)
    #                 plt.close()
    #                 print(f"\033[92m✔ salvato {output_png}\033[0m")
    #     except Exception as exc:
    #         print(f"⚠ Errore nella generazione dei grafici raw: {exc}")
    #     finally:
    #         # Remove temporary raw file to save storage
    #         try:
    #             os.remove(raw_path)
    #         except OSError:
    #             pass

    # --------------------------------------------------
    # Load z-score control data
    # --------------------------------------------------
    df = pd.read_csv(control_csv_path)

    t_raw = df["time"]
    if pd.api.types.is_numeric_dtype(t_raw) and t_raw.min() < 1e12:
        t = pd.date_range(
            start=f"{year}-{month}-01",
            periods=len(df),
            freq="15min",
        )[: len(df)]
    else:
        df["time"] = pd.to_datetime(df["time"])
        t = df["time"]

    # --------------------------------------------------
    # Plots + warnings/alarms (with labels)
    # --------------------------------------------------
    summary: list[dict] = []

    for col in df.columns:
        if col == "time":
            continue

        y = df[col].to_numpy()

        base_id = col.split("_")[0].upper()
        if base_id not in label_dict:
            raise ValueError(f"ID sensore '{base_id}' non presente nel file label-id")
        label = label_dict[base_id]

        suffix = col.split("_")[-1]
        axis_tag = f"_{suffix}" if suffix in ["x", "y"] else ""
        pretty_name = f"{label}{axis_tag}"

        # ----------------------
        # z-score
        # ----------------------
        in_range = np.abs(y) <= 3
        out_range = ~in_range
        y_in = y.copy()
        y_out = y.copy()
        y_out[in_range] = np.nan

        plt.figure(figsize=(12, 7))
        plt.plot(t, y_in, "-", color="black", linewidth=0.8)
        plt.scatter(t, y_out, color="darkorange", s=12)

        plt.fill_between([min(t), max(t)], -3, +3, color=params["colors"]["base_blue"], alpha=0.15)
        plt.hlines([-3, +3], min(t), max(t), color=params["colors"]["dark_blue"], linestyle="--")

        plt.text(
            0.5,
            0.125,
            "Lower Control Limit",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
            color=params["colors"]["dark_blue"],
            fontsize=FONT_SIZE,
        )
        plt.text(
            0.5,
            0.875,
            "Upper Control Limit",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
            color=params["colors"]["dark_blue"],
            fontsize=FONT_SIZE,
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

        n_warning = int(np.sum(np.abs(y) > 3))

        # ----------------------
        # livello di allarme
        # ----------------------
        controller = EventsController(z=y, time=t, k=3, decay_rate=0.05, alert_th=3.0)
        alarm_series_arr, alarm_events = controller.run()

        plt.figure(figsize=(12, 8))
        plt.plot(t, alarm_series_arr, color="black", linewidth=1.5)
        plt.axhline(3.0, color=params["colors"]["dark_blue"], linestyle="--", linewidth=1.2)
        plt.fill_between(t, 0, 3.0, color=params["colors"]["dark_blue"], alpha=0.2)

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

        summary.append(
            {
                "sensor_id": col,
                "label": pretty_name,
                "warnings": n_warning,
                "alarms": n_alarm,
            }
        )

    summary_df = pd.DataFrame(summary)
    out_csv = os.path.join(fig_out, f"{year}_{month}_summary.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"\033[92m\n✔ salvato {out_csv}\033[0m")
    return fig_out


# ======================================================
# CLASS PER EVENTI
# ======================================================


class EventsController:
    def __init__(self, z, time, k=3, decay_rate=0.05, alert_th=3.0):
        self.z = z
        self.time = time
        self.k = k
        self.decay_rate = decay_rate
        self.alert_th = alert_th

    def run(self):
        alarm_val = 0.0
        alarm_series = []
        alarm_events = []

        for i in range(len(self.z)):
            if alarm_val > 3:
                print("ALLARME")
                alarm_val = 0
            else:
                p = 1 if self.warning(i) else 0
                alarm_val = alarm_val * (1 - float(self.decay_rate)) + p

            alarm_series.append(alarm_val)

            if self.emergency(alarm_val):
                alarm_events.append(
                    {
                        "type": "emergency",
                        "timestamp": self.time[i] if not hasattr(self.time, "iloc") else self.time.iloc[i],
                        "value": alarm_val,
                    }
                )
        return np.array(alarm_series), alarm_events

    def warning_above(self, i):
        return self.z[i] > self.k

    def warning_below(self, i):
        return self.z[i] < -self.k

    def warning(self, i):
        return self.warning_below(i) or self.warning_above(i)

    def emergency(self, alarm_val):
        return alarm_val >= self.alert_th


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        # Explicit args: control_csv year month [opera_key]
        run_data_plots(
            control_csv_path=sys.argv[1],
            year=sys.argv[2],
            month=sys.argv[3],
            opera_key=sys.argv[4] if len(sys.argv) > 4 else None,
        )
    else:
        # Config-based: read from config_report.yaml
        config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config_report.yaml"
        run_data_plots(config_path=config_path)
