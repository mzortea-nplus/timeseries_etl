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

OPERA_KEY = "P005"   # <-- cambia qui se serve

OPERA_DIR = OPERE[OPERA_KEY]
MONTH_TAG = f"{year}_{month}"

FIG_ROOT = "figures"
FIG_PATH = os.path.join(FIG_ROOT, OPERA_DIR, MONTH_TAG)
os.makedirs(FIG_PATH, exist_ok=True)

OUT_ROOT = "outputs"
OUT_PATH = os.path.join(OUT_ROOT, OPERA_DIR, MONTH_TAG)
os.makedirs(OUT_PATH, exist_ok=True)


# ======================================================
# DATABASE
# ======================================================

duckdb_path = os.path.join("data", f"{OPERE[OPERA_KEY]}.duckdb")
if not os.path.exists(duckdb_path):
    raise FileNotFoundError(f"Database non trovato: {duckdb_path}")

conn = duckdb.connect(duckdb_path)

df_raw = conn.sql(f"""
    SELECT *
    FROM main_staging.all_static
    WHERE date_trunc('month', time) = make_date({year}, {month}, 1)
""").df()

print(df_raw)

df = conn.sql(f"""
    SELECT *
    FROM control
    WHERE date_trunc('month', time) = make_date({year}, {month}, 1)
""").df()


# ======================================================
# LOAD LABEL-ID ASSOCIATION
# ======================================================

label_path = os.path.join("data", "label-id", f"{OPERA_KEY}_label-id.csv")

if not os.path.exists(label_path):
    raise FileNotFoundError(f"File label-id non trovato: {label_path}")

label_df = pd.read_csv(label_path, sep=None, engine="python")

if label_df.shape[1] < 2:
    raise ValueError(
        f"Il file {label_path} deve avere almeno 2 colonne (label, id)"
    )

# Pulizia stringhe
label_df.iloc[:, 0] = label_df.iloc[:, 0].astype(str).str.strip()
label_df.iloc[:, 1] = label_df.iloc[:, 1].astype(str).str.strip()

# Creo dizionario usando SOLO parte prima di "_" e tutto maiuscolo
label_dict = {
    row_id.split("_")[0].upper(): label
    for label, row_id in zip(label_df.iloc[:, 0], label_df.iloc[:, 1])
}

print(f"✔ Caricate {len(label_dict)} associazioni ID → Label")



# ======================================================
# PLOT DATI GREZZI CON TEMPERATURA DI RIFERIMENTO
# ======================================================

FONT_SIZE = 20

t = pd.to_datetime(df_raw["time"])

deg_to_mrad = np.pi / 180 * 1000

# Funzione per unità di misura
def get_ylabel(sensor_id):
    suffix = sensor_id.split("_")[-1]  # prende ultima parte dopo "_"

    mapping = {
        "t": "Temperatura [°C]",
        "e": "Estensione [mm]",
        "s": "Spostamento [mm]",
        "x": "Rotazione longitudinale [mrad]",
        "y": "Rotazione trasversale [mrad]",
    }

    return mapping.get(suffix, sensor_id)


sensor_id = [col for col in df_raw.columns if col not in ["time", "month"]]
temp_sensors = [col for col in sensor_id if col.endswith("_t")]
struct_sensors = [col for col in sensor_id if not col.endswith("_t")]

if len(temp_sensors) == 0:
    raise ValueError("Nessuna sonda di temperatura trovata nel dataframe")
temp_col = temp_sensors[0]                                          # prende la prima sonda trovata
temperature = df_raw[temp_col]
print(f"✔ Temperatura di riferimento: {temp_col}")

for sensor_id in struct_sensors:

    base_id = sensor_id.split("_")[0].upper()

    if base_id not in label_dict:
        raise ValueError(
            f"ID sensore '{base_id}' non presente nel file label-id"
        )

    label = label_dict[base_id]

    y = df_raw[sensor_id].copy()

    # ------------------------------------
    # Conversione rotazioni in mrad
    # ------------------------------------
    suffix = sensor_id.split("_")[-1]
    axis_tag = f"_{suffix}" if suffix in ["x", "y"] else ""

    if suffix in ["x", "y"]:
        y = y * deg_to_mrad

    fig, ax1 = plt.subplots(figsize=(12,5))

    # Asse sinistro
    ax1.plot(t, y, linewidth=1)
    ax1.set_xlabel(f"{year}_{month} [gg]", fontsize=FONT_SIZE)
    ax1.set_ylabel(get_ylabel(sensor_id), fontsize=FONT_SIZE)
    ax1.xaxis.set_major_locator(mdates.DayLocator(bymonthday=range(5, 32, 5)))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
    ax1.tick_params(axis='both', labelsize=FONT_SIZE)
    ax1.grid(True, linestyle='--')

    # Asse destro temperatura
    ax2 = ax1.twinx()
    ax2.plot(t, temperature, color='darkorange', linewidth=1.5)
    ax2.set_ylabel("Temperatura [°C]", color='darkorange', fontsize=FONT_SIZE)
    ax2.tick_params(axis='y', labelcolor='darkorange', labelsize=FONT_SIZE)

    # Titolo SOLO con label
    axis_desc = get_ylabel(sensor_id)
    temp_label = label_dict[temp_col.split("_")[0].upper()]

    plt.title(f"{label}{axis_tag} e {temp_label}", fontsize=FONT_SIZE)

    plt.tight_layout()

    output_png = os.path.join(FIG_PATH, f"raw_{label}{axis_tag}.png")
    plt.savefig(output_png, dpi=300)
    plt.close()

    print(f"\033[92m✔ salvato {output_png}\033[0m")



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
                print("Alarm!")
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

    label = label_dict[base_id]

    # ----------------------
    # z-score
    # ----------------------
    in_range = np.abs(y) <= 3
    out_range = ~in_range
    y_in = y.copy()
    y_out = y.copy()
    y_out[in_range] = np.nan

    plt.figure(figsize=(12,7))
    plt.plot(t, y_in, '-', color='black', alpha=1.0, linewidth=0.8)
    plt.plot(t, y_out, '.-', color='orange', markersize=6)

    plt.fill_between([min(t), max(t)], -3, +3, color='green', alpha=0.20)
    plt.hlines([-3, +3], min(t), max(t), color='darkgreen', linestyle='--')

    plt.text(0.5, 0.125, 'Lower Control Limit', ha='center', va='center',
             transform=plt.gca().transAxes, color='darkgreen', fontsize=FONT_SIZE)
    plt.text(0.5, 0.875, 'Upper Control Limit', ha='center', va='center',
             transform=plt.gca().transAxes, color='darkgreen', fontsize=FONT_SIZE)

    ax = plt.gca()
    ax.set_xlabel(f"{year}_{month} [gg]", fontsize=FONT_SIZE)
    ax.set_ylabel("z-score", fontsize=FONT_SIZE)
    ax.tick_params(axis='both', labelrotation=0, labelsize=FONT_SIZE)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
    ax.set_yticks([-4,-3,-2,-1,0,1,2,3,4])
    ax.set_ylim([-5,+5])
    
    # Titolo SOLO con label
    base_id = col.split("_")[0].upper()
    if base_id not in label_dict:
        raise ValueError(
            f"ID sensore '{base_id}' non presente nel file label-id"
        )
    label = label_dict[base_id]
    suffix = col.split("_")[-1]
    axis_tag = f"_{suffix}" if suffix in ["x", "y"] else ""
    ax.set_title(f"{label}{axis_tag}", fontsize=FONT_SIZE)

    ax.grid(True, which='major', axis='both', linestyle='--')
    plt.tight_layout()

    output_png = os.path.join(FIG_PATH, f"z-score_{col}.png")
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"\033[92m✔ salvato {output_png}\033[0m")

    n_warning = np.sum(np.abs(y) > 3)

    # ----------------------
    # livello di allarme
    # ----------------------
    controller = EventsController(z=y, time=t, k=3, decay_rate=0.05, alert_th=3.0)
    alarm_series, alarm_events = controller.run()

    plt.figure(figsize=(12,8))
    plt.plot(t, alarm_series, color='black', linewidth=1.5)
    plt.axhline(3.0, color='green', linestyle='--', linewidth=1.2)
    plt.fill_between(t, 0, 3.0, color='green', alpha=0.2)

    ax = plt.gca()
    ax.set_xlabel(f"{year}_{month} [gg]", fontsize=FONT_SIZE)
    ax.set_ylabel("Livello di allerta", fontsize=FONT_SIZE)
    ax.set_title(f"Livello di allerta {col}", fontsize=FONT_SIZE)
    ax.tick_params(axis='x', labelrotation=0, labelsize=FONT_SIZE)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
    ax.grid(True, linestyle='--')
    plt.tight_layout()

    output_alert_png = os.path.join(FIG_PATH, f"alarm_{col}.png")
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
out_csv = os.path.join(OUT_PATH, f"summary_table.csv")
summary_df.to_csv(out_csv, index=False)
print(f"\033[92m\n✔ salvato {out_csv}\033[0m")

conn.close()
