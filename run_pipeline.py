#!/usr/bin/env python3
"""
Report automation pipeline:
  1. data_preparation.py - load config, clean & preprocess data from S3
  2. data_plots.py - produce z-score and alert plots
  3. torte_plots.py - produce torte-style pie charts (using same control data)
  4. report.py - produce the docx file
"""
import argparse
import os
import sys

import yaml
from data_plots import run_data_plots
from data_preparation import run_preparation
from report import run_report



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
    parser.add_argument(
        "--skip-data-prep",
        action="store_true",
        help="Skip step 1 (use existing control CSV)",
    )
    parser.add_argument(
        "--skip-torte",
        action="store_true",
        help="Skip step 3 (torte plots)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    site_code = config["site"]["code"]
    ym = config["data"]["month"]  # e.g. "2026-02"
    year, month = ym[:4], ym[5:7]
    control_csv = f"data/control_{site_code}_{year}_{month}.csv"

    # Opere mapping for paths
    opere = {
        "P001": "P001_Sommacampagna",
        "P002": "P002_Giuliari_Milani",
        "P003": "P003_Gua",
        "P004": "P004_Adige_Est",
        "P005": "P005_Adige_Ovest",
    }
    opera_dir = opere.get(site_code, f"{site_code}_Unknown")
    fig_dir = os.path.join("figures", opera_dir, f"{year}_{month}")

    print("\n" + "=" * 60)
    print("REPORT PIPELINE")
    print("=" * 60)

    print("!!! Warning: missing data is counted from aggregated values (SQL Query) !!!")

    # Step 1: Data preparation
    if not args.skip_data_prep:
        print("\n[1/4] Running data_preparation.py ...")
        run_preparation(
            config_path=args.config,
            key_id=args.key_id,
            secret=args.secret,
            output_path=control_csv
        )
    else:
        print("\n[1/4] Skipping data preparation (--skip-data-prep)")
        if not os.path.exists(control_csv):
            print(f"\033[91mError: {control_csv} not found. Run without --skip-data-prep.\033[0m")
            sys.exit(1)

    # Step 2: Alert dynamics
    print("\n[2/4] Running data_plots.py ...")

    run_data_plots(
        control_csv_path=control_csv,
        year=year,
        month=month,
        opera_key=site_code,
    )

    # Step 4: Report
    print("\n[4/4] Running report.py ...")

    run_report(config_path=args.config, year=year, month=month)
    print("\n" + "=" * 60)
    print("Pipeline completed successfully")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
