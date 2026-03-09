import duckdb


def start_s3_connection(key_id, secret):
    conn = duckdb.connect(":memory:")
    conn.execute("INSTALL httpfs")
    conn.execute("LOAD httpfs")
    conn.execute(f"""
        CREATE OR REPLACE SECRET s3_secret (
            TYPE s3,
            PROVIDER config,
            KEY_ID '{key_id}',
            SECRET '{secret}',
            REGION 'eu-central-1'
        )
    """)
    return conn


def retrieve_raw_data(filepath: str, conn):
    print(filepath)
    query = (
        f"""
        SELECT
            time_bucket(INTERVAL '15m', datetime) AS time_range,
            AVG(COLUMNS(*))
        FROM read_parquet('{filepath}')
        GROUP BY time_range
        ORDER BY time_range;
    """
        if "rot" in filepath
        else f"SELECT * from read_parquet('{filepath}') ORDER BY datetime"
    )
    if filepath is None:
        return None
    df = conn.execute(query).df()
    df.drop(columns=["datetime"], inplace=True)
    df = df.rename({"time_range": "datetime"})
    print(df.head(5))
    return df
