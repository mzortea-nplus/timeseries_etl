#!/usr/bin/env python3
"""
Report automation pipeline (backwards-compatible wrapper).

Calls the timeseries-etl package: prepare -> plot -> report.

Usage:
  python run_pipeline.py [--config PATH] [--skip-data-prep] [--skip-torte] [--key-id KEY] [--secret SECRET]

Equivalently:
  timeseries-etl run [same args]
"""

import sys
from timeseries_etl.cli import main

if __name__ == "__main__":
    # Inject "run" as subcommand so argparse sees: run_pipeline.py run --config ... --skip-data-prep ...
    argv = sys.argv[1:]
    if argv and argv[0] not in ("run", "prepare", "plot", "report", "agent"):
        sys.argv = [sys.argv[0], "run"] + argv
    sys.exit(main())
