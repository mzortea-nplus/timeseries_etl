import numpy as np 
import pandas as pd 
import duckdb
from matplotlib import pyplot as plt
import os 

os.makedirs("figures", exist_ok=True)

conn = duckdb.connect('dbt/dev.duckdb')

df = conn.sql("""
    SELECT * from control
""").df()

warnings = []
for col in df.columns:
    if col != 'time':
        n_warnings = (df[col].abs() > 3).sum()
        warnings.append({
            "sensor": col,
            "warnings": n_warnings
        })

        plt.plot(df['time'], df[col])
        plt.hlines([3, 3], min(df['time']), max(df['time']), linestyle='--')
        plt.hlines([-3, -3], min(df['time']), max(df['time']), linestyle='--')
        plt.savefig(f'figures/sensor_{col}.png')
        plt.close()

pd.DataFrame(warnings).to_csv("outputs/summary_table.csv", index=False)
conn.close()