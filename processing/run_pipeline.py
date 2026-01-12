import pandas as pd
from data_handling import AccelerometerDataOps, DuckDBOps, DataOps
from datetime import datetime, timedelta
import pytz
import time, psutil

p = psutil.Process()
t0 = time.time()

INTERVAL_MINUTES = 60 * 24 * 365
FS = {"acc": 200, "spost": 1 / (15 * 60), "tmp": 1 / (15 * 60)}
TARGET_FS = 50
FILTER_F = 50
PATH = "C:/Users/m.zortea/Documents/timeseries_etl/data/raw/"

# end_date = datetime.now(pytz.timezone("UTC"))
# start_date = end_date - timedelta(minutes=INTERVAL_MINUTES)
start_date = datetime(2025, 4, 1, tzinfo=pytz.utc)
end_date = datetime(2025, 4, 2, tzinfo=pytz.utc)

current_day = start_date
while current_day <= end_date:
    for data_type in ["acc"]:
        # ------------------------------------------------------------------
        # Initialize operators
        # ------------------------------------------------------------------
        operator = DataOps()

        # ------------------------------------------------------------------
        # Load raw data
        # ------------------------------------------------------------------
        df = operator.read_files(
            path=PATH + str(data_type),
            phm_list=None,
            start_date=current_day,
            end_date=current_day + timedelta(days=1),
        )
        fs = FS[data_type]
        # ------------------------------------------------------------------
        # Basic preprocessing (ordering, interpolation, de-meaning)
        # ------------------------------------------------------------------
        df = operator.basic_preprocessing(df)

        # ------------------------------------------------------------------
        # Split into continuous segments
        # ------------------------------------------------------------------
        if data_type != "acc":
            final_df = df.copy()
        else:
            accel_operator = AccelerometerDataOps()

            df_segments = accel_operator.split_segments(
                df,
                fs,
                min_segment_length_minutes=1,
            )

            # ------------------------------------------------------------------
            # Process and ingest each segment
            # ------------------------------------------------------------------
            processed_segments = []

            for segment_idx, segment_df in enumerate(df_segments):
                if data_type == "acc":
                    processed_df = accel_operator.data_preprocessing(
                        df=segment_df,
                        fs=fs,
                        filter_fs=FILTER_F,
                        target_fs=TARGET_FS,
                    )
                else:
                    processed_df = segment_df.copy()

                processed_segments.append(processed_df)

            final_df = pd.concat(processed_segments, ignore_index=True)

        final_df.to_parquet(
            path=f"data/processed/{data_type}_segmented.parquet",
            index=False,
        )
        current_day += timedelta(days=2)

# model logic

print(f"METRIC runtime_seconds={time.time() - t0}")
print(f"METRIC rss_Mbytes={p.memory_info().rss/1024/1024}")
