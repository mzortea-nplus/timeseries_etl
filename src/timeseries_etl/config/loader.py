"""Load and normalize YAML configuration."""

from pathlib import Path

import yaml


def load_config(config_path: str | Path) -> dict:
    """Load report configuration from YAML file."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_month_year(ym: str) -> tuple[str, str]:
    """Extract year and month from config value (YYYY-MM or YYYY-MM-DD)."""
    parts = ym.strip().split("-")
    year = parts[0] if len(parts) >= 1 else ""
    month = parts[1] if len(parts) >= 2 else ""
    return year, month


def build_data_paths(config: dict) -> dict[str, str | None]:
    """Build S3 parquet glob paths from config (str_path, tmp_path, incl_path, spost_path, month)."""
    data_cfg = config["data"]
    site_code = config["site"]["code"]
    ym = data_cfg["month"]

    incl_pattern = None
    spost_pattern = None
    str_pattern = None
    tmp_pattern = None

    if "incl_path" in data_cfg:
        incl_path = data_cfg["incl_path"].rstrip("/")
        incl_pattern = f"{incl_path}/{site_code}_{ym}*_rot.parquet"
    if "spost_path" in data_cfg:
        spost_path = data_cfg["spost_path"].rstrip("/")
        spost_pattern = f"{spost_path}/{site_code}_{ym}*_dsp.parquet"
    if "str_path" in data_cfg:
        str_path = data_cfg["str_path"].rstrip("/")
        str_pattern = f"{str_path}/{site_code}_{ym}*_str.parquet"
    if "tmp_path" in data_cfg:
        tmp_path = data_cfg["tmp_path"].rstrip("/")
        tmp_pattern = f"{tmp_path}/{site_code}_{ym}*_tmp.parquet"

    return {
        "incl": incl_pattern,
        "spost": spost_pattern,
        "str": str_pattern,
        "tmp": tmp_pattern,
    }
