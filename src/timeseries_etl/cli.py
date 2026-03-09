"""CLI entrypoint with subcommands run, prepare, plot, report, agent."""

import argparse
import os
import sys


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default="configs/config_report.yaml",
        help="Path to config YAML",
    )
    parser.add_argument("--key-id", help="AWS access key (or set AWS_ACCESS_KEY_ID env)")
    parser.add_argument(
        "--secret",
        help="AWS secret key (or set AWS_SECRET_ACCESS_KEY env)",
    )


def cmd_run(args: argparse.Namespace) -> int:
    """Run full pipeline."""
    from timeseries_etl.pipeline.orchestrate import run_full_pipeline

    run_full_pipeline(
        config_path=args.config,
        key_id=args.key_id or os.environ.get("AWS_ACCESS_KEY_ID"),
        secret=args.secret or os.environ.get("AWS_SECRET_ACCESS_KEY"),
        skip_data_prep=args.skip_data_prep,
    )
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    """Run data preparation only."""
    from timeseries_etl.pipeline.prepare import run_prepare

    run_prepare(
        config_path=args.config,
        key_id=args.key_id or os.environ.get("AWS_ACCESS_KEY_ID"),
        secret=args.secret or os.environ.get("AWS_SECRET_ACCESS_KEY"),
        output_path=args.output,
    )
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    """Run plotting only."""
    from timeseries_etl.config import load_config, normalize_month_year
    from timeseries_etl.pipeline.plot import run_plot

    config = load_config(args.config)
    site_code = config["site"]["code"]
    ym = config["data"]["month"]
    year, month = normalize_month_year(ym)
    control_csv = args.control_csv or f"data/control_{site_code}_{year}_{month}.csv"

    run_plot(
        control_csv_path=control_csv,
        year=args.year or year,
        month=args.month or month,
        opera_key=args.opera_key or site_code,
        config_path=args.config,
    )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Run report generation only."""
    from timeseries_etl.config import load_config, normalize_month_year
    from timeseries_etl.pipeline.report import run_report

    config = load_config(args.config)
    ym = config["data"]["month"]
    year, month = normalize_month_year(ym)

    run_report(
        config_path=args.config,
        year=args.year or year,
        month=args.month or month,
    )
    return 0


def cmd_agent(args: argparse.Namespace) -> int:
    """Placeholder for future LLM agent subcommand."""
    print("timeseries-etl agent: not implemented yet. Reserved for future local LLM agent.")
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """Validate config and output layout without AWS access."""
    from pathlib import Path

    from timeseries_etl.config import build_data_paths, load_config, normalize_month_year

    config_path = getattr(args, "config", "configs/config_report.yaml") or "configs/config_report.yaml"
    print("Validating config and layout (no AWS access)...")

    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        print(f"  [FAIL] Config not found: {e}")
        return 1
    except Exception as e:
        print(f"  [FAIL] Invalid config: {e}")
        return 1
    print(f"  [OK] Config loaded: {config_path}")

    site_code = config["site"]["code"]
    ym = config["data"]["month"]
    year, month = normalize_month_year(ym)
    paths = build_data_paths(config)
    control_csv = Path(f"data/control_{site_code}_{year}_{month}.csv")
    fig_dir = Path("figures") / site_code / f"{year}_{month}"
    out_dir = Path("outputs") / site_code / f"{year}_{month}"

    print(f"  Site: {site_code}")
    print(f"  Period: {year}-{month}")
    print(f"  Control CSV: {control_csv}")
    print(f"  Figures dir: {fig_dir}")
    print(f"  Output dir: {out_dir}")
    print(f"  Data patterns: {list(paths.keys())}")

    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    print("  [OK] Output directories created (or exist)")

    print("\nSmoke check OK. Run full pipeline with AWS credentials for data fetch.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="timeseries-etl",
        description="Timeseries ETL: sensor data preparation, plotting, report generation",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = subparsers.add_parser("run", help="Run full pipeline")
    _add_common_args(p_run)
    p_run.add_argument("--skip-data-prep", action="store_true", help="Skip data preparation")
    p_run.set_defaults(func=cmd_run)

    # prepare
    p_prep = subparsers.add_parser("prepare", help="Run data preparation only")
    _add_common_args(p_prep)
    p_prep.add_argument("--output", "-o", help="Output control CSV path")
    p_prep.set_defaults(func=cmd_prepare)

    # plot
    p_plot = subparsers.add_parser("plot", help="Run plotting only")
    _add_common_args(p_plot)
    p_plot.add_argument("--control-csv", help="Path to control CSV")
    p_plot.add_argument("--year", help="Year (YYYY)")
    p_plot.add_argument("--month", help="Month (MM)")
    p_plot.add_argument("--opera-key", help="Opera/site code (e.g. P001)")
    p_plot.set_defaults(func=cmd_plot)

    # report
    p_report = subparsers.add_parser("report", help="Run report generation only")
    _add_common_args(p_report)
    p_report.add_argument("--year", help="Year (YYYY)")
    p_report.add_argument("--month", help="Month (MM)")
    p_report.set_defaults(func=cmd_report)

    # agent (placeholder)
    p_agent = subparsers.add_parser("agent", help="Local LLM agent (placeholder)")
    p_agent.set_defaults(func=cmd_agent)

    # smoke (validate config and layout without AWS)
    p_smoke = subparsers.add_parser("smoke", help="Validate config and create output dirs (no AWS)")
    p_smoke.add_argument("--config", default="configs/config_report.yaml", help="Path to config YAML")
    p_smoke.set_defaults(func=cmd_smoke)

    args = parser.parse_args()
    try:
        return args.func(args)
    except NotImplementedError as e:
        print(f"\033[91mError: {e}\033[0m", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
