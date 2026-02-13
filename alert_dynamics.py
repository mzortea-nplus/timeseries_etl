import numpy as np 
import pandas as pd 
import duckdb
import os 
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
import sys

# ======================================================
# INPUT DATA
# ======================================================

my_date = sys.argv[1]          # es: 2026-01
year, month = my_date.split('-')

# ======================================================
# CARTELLE OUTPUT
# ======================================================

OPERE = {
    "P001": "P001_Sommacampagna",
    "P002": "P002_Giuliari_Milani",
    "P003": "P003_Gua",
    "P004": "P004_Adige_Est",
    "P005": "P005_Adige_Ovest",
}

OPERA_KEY = "P001"   # <-- cambia qui se serve

FIG_ROOT = "figures"
OPERA_DIR = OPERE[OPERA_KEY]
MONTH_TAG = f"{year}_{month}"

FIG_OUT = os.path.join(FIG_ROOT, OPERA_DIR, MONTH_TAG)
os.makedirs(FIG_OUT, exist_ok=True)

# ======================================================
# DATABASE
# ======================================================

duckdb_path = os.path.join("data", f"{OPERE[OPERA_KEY]}.duckdb")
if not os.path.exists(duckdb_path):
    raise FileNotFoundError(f"Database non trovato: {duckdb_path}")

conn = duckdb.connect(duckdb_path)

df = conn.sql(f"""
    SELECT *
    FROM control
    WHERE date_trunc('month', time) = make_date({year}, {month}, 1)
""").df()

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
                print("ALLARMEEEEEEE")
                alarm_val = 0
            else:
                p = 1 if self.warning(i) else 0
                alarm_val = alarm_val * (1 - float(self.decay_rate)) + p

            alarm_series.append(alarm_val)

            if self.emergency(alarm_val):
                alarm_events.append({
                    "type": "emergency",
                    "timestamp": self.time[i],
                    "value": alarm_val
                })
        return np.array(alarm_series), alarm_events

    def warning_above(self, i):
        return self.z[i] > self.k

    def warning_below(self, i):
        return self.z[i] < -self.k

    def warning(self, i):
        return self.warning_below(i) or self.warning_above(i)

    def emergency(self, alarm_val):
        return alarm_val >= self.alert_th

# ======================================================
# PLOT + CALCOLO WARNINGS E ALLARMI
# ======================================================

summary = []

for col in df.columns:
    if col == "time":
        continue

    y = df[col].to_numpy()
    t = df["time"]

    # ----------------------
    # z-score
    # ----------------------
    in_range = np.abs(y) <= 3
    out_range = ~in_range
    y_in = y.copy()
    y_out = y.copy()
    y_out[in_range] = np.nan

    plt.figure(figsize=(12,5))
    plt.plot(t, y_in, '-', color='black', alpha=0.5, linewidth=0.8)
    plt.plot(t, y_out, 'r.-', markersize=4)

    plt.fill_between([min(t), max(t)], -3, +3, color='skyblue', alpha=0.35)
    plt.hlines([-3, +3], min(t), max(t), color='blue', linestyle='--')

    plt.text(0.5, 0.125, 'Lower Control Limit', ha='center', va='center',
             transform=plt.gca().transAxes, color='blue')
    plt.text(0.5, 0.875, 'Upper Control Limit', ha='center', va='center',
             transform=plt.gca().transAxes, color='blue')

    ax = plt.gca()
    ax.set_xlabel(f"{year}_{month} [gg]")
    ax.set_ylabel("z-score")
    ax.tick_params(axis='x', labelrotation=0)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
    ax.set_yticks([-4,-3,-2,-1,0,1,2,3,4])
    ax.set_ylim([-5,+5])
    ax.set_title(f"z-score {col}")
    ax.grid(True, which='major', axis='both', linestyle='--')
    plt.tight_layout()

    output_png = os.path.join(FIG_OUT, f"sensor_{col}.png")
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"\033[92m✔ salvato {output_png}\033[0m")

    n_warning = np.sum(np.abs(y) > 3)

    # ----------------------
    # livello di allarme
    # ----------------------
    controller = EventsController(z=y, time=t, k=3, decay_rate=0.05, alert_th=3.0)
    alarm_series, alarm_events = controller.run()

    plt.figure(figsize=(12,5))
    plt.plot(t, alarm_series, color='black', linewidth=1.5)
    plt.axhline(3.0, color='green', linestyle='--', linewidth=1.2)
    plt.fill_between(t, 0, 3.0, color='green', alpha=0.2)

    ax = plt.gca()
    ax.set_xlabel(f"{year}_{month} [gg]")
    ax.set_ylabel("Livello di allerta")
    ax.set_title(f"Livello di allerta {col}")
    ax.tick_params(axis='x', labelrotation=0)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
    ax.grid(True, linestyle='--')
    plt.tight_layout()

    output_alert_png = os.path.join(FIG_OUT, f"alarm_{col}.png")
    plt.savefig(output_alert_png, dpi=300)
    plt.close()
    print(f"\033[92m✔ salvato {output_alert_png}\033[0m")

    n_alarm = np.sum(alarm_series >= 3.0)

    summary.append({
        "sensor": col,
        "warnings": int(n_warning),
        "alarms": int(n_alarm)
    })

# ======================================================
# CSV SUMMARY
# ======================================================

summary_df = pd.DataFrame(summary)
out_csv = os.path.join(FIG_OUT, f"{year}_{month}_summary.csv")
summary_df.to_csv(out_csv, index=False)
print(f"\033[92m\n✔ salvato {out_csv}\033[0m")

conn.close()
