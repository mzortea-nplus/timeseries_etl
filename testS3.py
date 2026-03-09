import argparse
import os
import sys
import duckdb

def load_config(config_path: str = "configs/config_report.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Run the full report pipeline")
    parser.add_argument(
        "--config",
        default="configs/config_report.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--key-id",
        help="AWS access key (or set AWS_ACCESS_KEY_ID env)",
    )
    parser.add_argument(
        "--secret",
        help="AWS secret key (or set AWS_SECRET_ACCESS_KEY env)",
    )
    args = parser.parse_args()

    duckdb.sql("INSTALL httpfs")
    duckdb.sql("LOAD httpfs")
    duckdb.sql(f"""
        CREATE OR REPLACE SECRET s3_secret (
            TYPE s3,
            PROVIDER config,
            KEY_ID '{args.key_id}',
            SECRET '{args.secret}',
            REGION 'eu-central-1'
        )
    """
    )

    query = f"""
        SELECT
            any_value(COLUMNS(*))
        FROM read_parquet('s3://vittorio-field-data/field_data/A4/P003/rot/parquet/P003_2026-03-05_09-00-00_rot.parquet')      
    """
    print(duckdb.sql(query).df())


if __name__ == '__main__':
    main()