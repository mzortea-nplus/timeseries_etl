import os
import re
from datetime import datetime
import duckdb
import pandas as pd
from scipy import signal
import numpy as np
from typing import Optional
import pytz


# ============================================================
# DuckDB Utilities
# ============================================================
class DuckDBOps:
    @staticmethod
    def from_df_to_DuckDB(df: pd.DataFrame):
        pass


# ============================================================
# Generic data utilities
# ============================================================
class DataOps:
    @staticmethod
    def extract_date_from_filename(filename: str) -> datetime:
        match = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}", filename)
        if not match:
            raise ValueError(f"No valid date found in filename: {filename}")
        return datetime.strptime(match.group(0), "%Y-%m-%d_%H").astimezone(pytz.utc)

    @staticmethod
    def basic_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
        if "time" not in df.columns:
            raise ValueError("Missing 'time' column")
        df = df.sort_values("time").reset_index(drop=True)
        num_cols = df.columns.drop("time")
        df[num_cols] = df[num_cols] - np.nanmean(df[num_cols])
        return df

    @staticmethod
    def detect_holes(df: pd.DataFrame, fs: float):
        if len(df) < 2:
            return []

        expected_dt = 1.0 / fs

        dt = df["time"].diff().dt.total_seconds()
        hole_mask = dt > (1.1 * expected_dt)

        return np.flatnonzero(hole_mask.to_numpy())

    @staticmethod
    def read_files(
        path: str,
        phm_list=None,
        start_date=pd.to_datetime("2025-03-01"),
        end_date=pd.to_datetime("2025-03-02"),
    ) -> pd.DataFrame:

        full_df = []
        print(f"Searching files in path {path}")
        for fname in os.listdir(path):
            fpath = os.path.join(path, fname)
            if not os.path.isfile(fpath):
                continue

            file_date = DataOps.extract_date_from_filename(fname)

            if not (start_date <= file_date <= end_date):
                continue

            if fname.endswith(".parquet"):
                df = pd.read_parquet(fpath)
            elif fname.endswith(".csv"):
                df = pd.read_csv(fpath, delimiter=";")
            else:
                continue

            date_format = (
                "%Y/%m/%d %H:%M:%S:%f" if "acc" in path else "%Y/%m/%d %H:%M:%S"
            )
            df["time"] = pd.to_datetime(df["time"], format=date_format, errors="raise")

            full_df.append(df.dropna(subset=["time"]))

        if not full_df:
            raise ValueError("No matching data found")

        full_df = pd.concat(full_df, ignore_index=True)

        if phm_list is not None:
            missing = set(phm_list) - set(full_df.columns)
            if missing:
                raise ValueError(f"Missing columns: {missing}")
            full_df = full_df[["time"] + phm_list]

        nans_count = full_df.isna().sum()
        for col in full_df.columns:
            n = full_df.shape[0]
            if nans_count[col]:
                print(
                    f"\t - {col}: {nans_count[col]} invalid entries ({(nans_count[col] / n * 100):.2f} %)"
                )

        return full_df.sort_values("time").reset_index(drop=True)


# ============================================================
# Accelerometer-specific operations
# ============================================================


class AccelerometerDataOps(DataOps):
    @staticmethod
    def split_segments(
        df: pd.DataFrame,
        fs: float,
        min_segment_length_minutes: int = 15,
    ):
        df = df.sort_values("time").reset_index(drop=True)

        gap_indices = DataOps.detect_holes(df, fs)
        segments = []

        boundaries = [0] + gap_indices.tolist() + [len(df)]

        for i in range(len(boundaries) - 1):
            seg = df.iloc[boundaries[i] : boundaries[i + 1]]

            if len(seg) < 2:
                continue

            duration = seg["time"].iloc[-1] - seg["time"].iloc[0]
            if duration >= pd.Timedelta(minutes=min_segment_length_minutes):
                segments.append(seg.reset_index(drop=True))

        print("Divided dataset into segments:")
        print(
            f"\t - N. holes {len(gap_indices)} ({int(len(gap_indices) / df.shape[0] * 100)} %)"
        )
        print(f"\t - N. segments {len(segments)}")

        return segments

    @staticmethod
    def data_preprocessing(
        df: pd.DataFrame,
        fs: float,
        filter_fs: Optional[float] = None,
        target_fs: Optional[float] = None,
    ) -> pd.DataFrame:

        df = df.sort_values("time").reset_index(drop=True)
        t0, t1 = df["time"].iloc[0], df["time"].iloc[-1]

        target_fs = target_fs or fs
        cutoff = filter_fs if filter_fs is not None else target_fs / 2.0 * 0.9
        if cutoff <= 0 or cutoff >= fs / 2:
            raise ValueError("Invalid filter cutoff frequency")

        # ------------------------------------------------------------------
        # Uniform grid
        # ------------------------------------------------------------------
        uniform_time = pd.date_range(
            start=t0,
            end=t1,
            freq=pd.to_timedelta(1 / fs, unit="s"),
        )

        df_uniform = (
            df.set_index("time")
            .reindex(uniform_time)
            .interpolate(method="time")
            .ffill()
            .bfill()
            .reset_index()
            .rename(columns={"index": "time"})
        )

        data = df_uniform.drop(columns="time").to_numpy()

        # ------------------------------------------------------------------
        # Anti-aliasing low-pass filter
        # ------------------------------------------------------------------
        sos = signal.iirfilter(
            N=6,
            Wn=cutoff,
            btype="lowpass",
            ftype="butter",
            fs=fs,
            output="sos",
        )
        data = signal.sosfiltfilt(sos, data, axis=0)

        # ------------------------------------------------------------------
        # Resampling
        # ------------------------------------------------------------------
        n_samples = int(len(data) * target_fs / fs)
        data = signal.resample(data, n_samples, axis=0)

        new_time = t0 + pd.to_timedelta(np.arange(n_samples) / target_fs, unit="s")

        out = pd.DataFrame(data, columns=df_uniform.columns.drop("time"))
        out.insert(0, "time", new_time)

        return out
