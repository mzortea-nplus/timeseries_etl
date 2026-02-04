import numpy as np 
import pandas as pd 
import duckdb
import os 
from matplotlib import pyplot as plt
from datetime import date, timedelta

os.makedirs("figures", exist_ok=True)

conn = duckdb.connect('dbt/etl.duckdb')

df = conn.sql("""
    SELECT * from control
    where date_trunc('month', time) between '2025-10-1' and '2025-10-31'
""").df()

warnings = []
for col in df.columns:
    if col != 'time':

        y = df[col].to_numpy()
        t = df['time']

        in_range = np.abs(y) <= 3
        out_range = ~in_range

        y_in = y.copy()
        y_out = y.copy()

        #y_in[out_range] = np.nan
        y_out[in_range] = np.nan

        plt.plot(t, y_in, '-', color='black', alpha=0.6, linewidth=0.8)
        plt.plot(t, y_out, 'r.-', markersize=4)
        plt.fill_between([min(t), max(t)], -3, +3, color='skyblue', alpha=0.65)
        plt.hlines([-3, -3], min(t), max(t), color='blue', linestyle='--')
        plt.hlines([+3, +3], min(t), max(t), color='blue', linestyle='--')
        plt.text(0.5, 0.125, 'Lower Control Limit', horizontalalignment='center',
        verticalalignment='center', transform=plt.gca().transAxes, color='blue')
        plt.text(0.5, 0.875, 'Upper Control Limit', horizontalalignment='center',
        verticalalignment='center', transform=plt.gca().transAxes, color='blue')
        

        #plt.yticks([-3, 3])
        plt.yticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])

        import matplotlib.dates as mdates
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%a %d'))

        #plt.xlim([min(t) - timedelta(days=10), max(t) + timedelta(days=10)])
        plt.ylim([-5, +5])
        plt.title(f"z-score {col}")
        plt.xticks(rotation=30)
        plt.grid(linestyle='--')
        plt.tight_layout()
        plt.savefig(f'figures/sensor_{col}.png')
        plt.close()

pd.DataFrame(warnings).to_csv("outputs/summary_table.csv", index=False)
conn.close()