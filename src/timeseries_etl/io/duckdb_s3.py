"""DuckDB httpfs setup and S3 parquet retrieval."""

import duckdb
import pandas as pd


def start_s3_connection(key_id: str, secret: str) -> None:
    """Configure DuckDB httpfs for S3 access."""
    duckdb.sql("INSTALL httpfs")
    duckdb.sql("LOAD httpfs")
    duckdb.sql(
        f"""
        CREATE OR REPLACE SECRET s3_secret (
            TYPE s3,
            PROVIDER config,
            KEY_ID '{key_id}',
            SECRET '{secret}',
            REGION 'eu-central-1'
        )
        """
    )


def retrieve_raw_data(filepath: str) -> pd.DataFrame:
    """Read parquet from S3, optionally aggregate rot files by 15m bucket."""
    print(filepath)

    query = f"SELECT any_value(COLUMNS(*)) FROM read_parquet('{filepath}')"
    df = duckdb.sql(query).df()

    df = df.dropna(axis=1, how="all")
    columns_to_query = df.columns.tolist()
    cols = [c for c in columns_to_query if c != "datetime"]

    if "rot" in filepath:
        agg_cols = [
            f'avg(CAST("{c}" AS DOUBLE)) AS "{c}"' for c in cols
        ]
        cols_sql = ", ".join(agg_cols)
        query = f"""
            SELECT
                time_bucket(INTERVAL '15m', datetime) AS time_range,
                {cols_sql}
            FROM read_parquet('{filepath}')
            GROUP BY time_range
            ORDER BY time_range
        """
        df = duckdb.sql(query).df()
        df = df.rename(columns={"time_range": "datetime"})
    else:
        agg_cols = [f'CAST("{c}" AS DOUBLE) AS "{c}"' for c in cols]
        cols_sql = ", ".join(agg_cols)
        query = f"SELECT datetime, {cols_sql} FROM read_parquet('{filepath}')"
        df = duckdb.sql(query).df()

    print(df.head(5))
    return df
