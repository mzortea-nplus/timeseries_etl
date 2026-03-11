"""Orchestrate full pipeline (prepare -> plot -> report)."""

import os

from timeseries_etl.config import load_config, normalize_month_year


def run_full_pipeline(
    config_path: str = "configs/config_report.yaml",
    key_id: str | None = None,
    secret: str | None = None,
    skip_data_prep: bool = False,
) -> None:
    """Run prepare, plot, report steps in order."""
    from timeseries_etl.pipeline.prepare import run_prepare
    from timeseries_etl.pipeline.plot import run_plot
    from timeseries_etl.pipeline.report import run_report

    config = load_config(config_path)
    site_code = config["site"]["code"]
    ym = config["data"]["month"]
    year, month = normalize_month_year(ym)
    control_csv = f"data/control_{site_code}_{year}_{month}.csv"

    print("\n" + "=" * 60)
    print("REPORT PIPELINE")
    print("=" * 60)
    print("!!! Warning: missing data is counted from aggregated values (SQL Query) !!!")

    if not skip_data_prep:
        print("\n[1/4] Running data preparation ...")
        run_prepare(
            config_path=config_path,
            key_id=key_id,
            secret=secret,
            output_path=control_csv,
        )
    else:
        print("\n[1/4] Skipping data preparation (--skip-data-prep)")
        if not os.path.exists(control_csv):
            print(
                f"\033[91mError: {control_csv} not found. Run without --skip-data-prep.\033[0m"
            )
            raise SystemExit(1)

    print("\n[2/4] Running data plots ...")
    run_plot(
        control_csv_path=control_csv,
        year=year,
        month=month,
        opera_key=site_code,
    )

    print("\n[3/3] Running report ...")
    run_report(config_path=config_path, year=year, month=month)

    print("\n" + "=" * 60)
    print("Pipeline completed successfully")
    print("=" * 60 + "\n")
