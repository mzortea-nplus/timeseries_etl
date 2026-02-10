import pandas as pd

from training import training
from prediction import predict_ts
from pipeline import TrainingPipeline

import pickle
import numpy as np
import sys
import duckdb
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
import yaml
import os

import mlflow

with open(sys.argv[1], "r") as f:
    config = yaml.safe_load(f)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("sindy training")

# Test logging
mlflow.start_run(run_name="LinReg")
mlflow.log_param("regressor", config["sindy"]["regressor"])
mlflow.log_param("fourier", config["sindy"]["fourier"])
mlflow.log_param("Site", "Sommacampagna")


def plot_sensor(t, measured, predicted):
    plt.plot(t, measured, color="blue", label="measured")
    plt.plot(t, predicted, color="red", label="predicted")
    plt.legend()
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))

    plt.xticks(rotation=30)
    plt.grid(linestyle="--")
    plt.tight_layout()
    plt.savefig(f"figures/sindy/sensor_{s}.png")
    plt.close()


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

    # find temperature sensors
    tmp_sensors = [s for s in df.columns if s.endswith("_t")]

    if config["mode"] == "training":
        # print(df.head())
        models = training(
            df,
            training_pipeline=TrainingPipeline(
                [el for el in config["sindy"]["fourier"]],
                alpha=2e-3,
                regressor=config["sindy"]["regressor"],
            ),
            debug=True,
        )
        os.makedirs(config["out"]["model"], exist_ok=True)
        for m in models:
            file_path = os.path.join(config["out"]["model"], m["sensor"] + ".pkl")
            with open(file_path, "wb") as f:
                pickle.dump(m["model"], f)

    elif config["mode"] == "prediction":
        os.makedirs("figures/sindy", exist_ok=True)
        available_models = os.listdir(config["out"]["model"])

        residuals_df = pd.DataFrame()
        for s in available_models:
            s = s.replace(".pkl", "")
            print("Sensor", s)
            if s not in df.columns:
                print("  -- skipping --")
                continue
            file_path = os.path.join(config["out"]["model"], s + ".pkl")
            with open(file_path, "rb") as f:
                model = pickle.load(f)
            predicted = predict_ts(
                df[[s] + tmp_sensors + ["time"]].rename(columns={s: "val"}), model
            )
            residuals = df[s].values - predicted
            residuals_df[s] = residuals
            plot_sensor(df["time"], df[s], predicted)

        residuals_df.to_parquet(config["out"]["residuals"])
    else:
        raise Exception(
            "In file sindy.yaml: 'mode' should be 'training' or 'prediction'."
        )

mlflow.end_run()
