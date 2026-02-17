from statsmodels.tsa.seasonal import STL
import sys
import duckdb
import yaml
import pandas as pd
from matplotlib import pyplot as plt

with open(sys.argv[1], "r") as f:
    config = yaml.safe_load(f)

if __name__ == "__main__":

    conn = duckdb.connect(database=config["data"]["db"], read_only=True)
    df = conn.execute(
        f"""
        SELECT * FROM main_staging.all_static
        WHERE time between '{config['data']['start_date']}' and '{config['data']['end_date']}'
    """
    ).df()

    # fix irregular sampling
    dts = df["time"].diff().dt.total_seconds()
    dt = dts.mode().iloc[0]
    mask = dts.isna() | (dts != dt)
    if any(mask[1:]):
        print("Found irregular sampling")
        print(df["time"][mask])
    df = df.loc[~mask]
    # df = df.set_index("time")

    dt = dts.mode().iloc[0]  # seconds per sample
    season_seconds = 24 * 3600  # example: daily seasonality
    period = int(season_seconds / dt)

    # find temperature sensors
    tmp_sensors = [s for s in df.columns if s.endswith("_t")]
    sensors = [
        c for c in df.columns if c not in ["dt", "time", "month", "tmp"] + tmp_sensors
    ]

    df[sensors] = df[sensors].fillna(value=df.mean())

    for s in sensors:
        res = STL(
            df[s].to_numpy(),
            period=period,
            seasonal=period * 365 + 1,
            trend=period + 1,
        ).fit()
        plt.plot(df["time"], res.trend, color="black", label="signal trend")
        plt.gca().twinx()
        plt.plot(
            df["time"], df[tmp_sensors].mean(axis=1), "--", label="avg tmp", alpha=0.5
        )
        plt.grid()
        plt.legend()
        plt.show()
