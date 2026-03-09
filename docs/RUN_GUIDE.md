# Timeseries ETL – Run Guide

## 1. Install and enter the environment

```bash
cd path/to/timeseries_etl
uv sync --extra dev   # install deps + pytest
```

Run commands via:

```bash
uv run <command>
```

---

## 2. Full pipeline

Reads site and month from `configs/config_report.yaml` and runs **prepare → plot → report**.

```bash
# Recommended CLI
uv run timeseries-etl run --config configs/config_report.yaml \
  --key-id YOUR_KEY --secret YOUR_SECRET

# Legacy-style wrapper (same effect)
uv run python run_pipeline.py --config configs/config_report.yaml \
  --key-id YOUR_KEY --secret YOUR_SECRET
```

Skip preparation (use existing control CSV):

```bash
uv run timeseries-etl run --config configs/config_report.yaml --skip-data-prep
```

---

## 3. Individual steps

### 3.1 Preparation only

Fetches data from S3 via DuckDB, runs thermal model/control, writes control CSV, raw-data plots, and `outputs/nans_percentage.csv`.

```bash
uv run timeseries-etl prepare \
  --config configs/config_report.yaml \
  --key-id YOUR_KEY --secret YOUR_SECRET \
  --output data/control_P005_2025_12.csv   # optional override
```

### 3.2 Plotting only

Consumes a control CSV and produces:
- z-score plots
- alert-level plots
- `<YYYY>_<MM>_summary.csv`

```bash
uv run timeseries-etl plot \
  --config configs/config_report.yaml \
  --control-csv data/control_P005_2025_12.csv \
  --year 2025 --month 12 --opera-key P005
```

Figures are written under:

```
figures/<opera_key>/<YYYY_MM>/
```

### 3.3 Report only

Consumes existing figures and CSVs, writes the DOCX report.

```bash
uv run timeseries-etl report \
  --config configs/config_report.yaml \
  --year 2025 --month 12
```

Output path:

```
outputs/<opera_key>/<YYYY_MM>/A4_<opera_key>_<YYYY_MM>.docx
```

---

## 4. Smoke check (no AWS)

Validate config and directory layout **without** S3 access:

```bash
uv run timeseries-etl smoke --config configs/config_report.yaml
```

This will:
- Load `configs/config_report.yaml`
- Print site code and period
- Ensure these directories exist: `figures/<site_code>/<YYYY_MM>/`, `outputs/<site_code>/<YYYY_MM>/`, `data/`, `outputs/`

---

## 5. Tests

```bash
uv run pytest tests/ -q
```
