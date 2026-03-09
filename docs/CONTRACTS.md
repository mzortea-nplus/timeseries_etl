# Public Contracts (Legacy Compatibility)

Contracts to preserve when migrating from `old_code/` to the new package.

## CLI Arguments (run_pipeline)

| Arg | Default | Description |
|-----|---------|-------------|
| `--config` | `configs/config_report.yaml` | Path to report config YAML |
| `--key-id` | env `AWS_ACCESS_KEY_ID` | AWS access key |
| `--secret` | env `AWS_SECRET_ACCESS_KEY` | AWS secret key |
| `--skip-data-prep` | false | Skip step 1, use existing control CSV |
| `--skip-torte` | false | Skip torte plots (step 3) |

## Config Keys (config_report.yaml)

- `site.name`, `site.code` (e.g. P001, P005)
- `data.str_path`, `data.tmp_path`, `data.incl_path`, `data.spost_path` (S3 parquet paths)
- `data.month` (YYYY-MM or YYYY-MM-DD; normalize to year/month)

## parameters.yaml

- `colors.base_blue`, `colors.dark_blue`, `colors.light_blue`

## Output Paths

| Path | Producer | Consumer |
|------|----------|----------|
| `data/control_{site}_{year}_{month}.csv` | prepare | plot, report |
| `data/label-id/{OPERA_KEY}_label-id.csv` | (input) | prepare, plot, report |
| `outputs/nans_percentage.csv` | prepare | report |
| `figures/{opera_dir}/{year}_{month}/` | prepare (raw), plot | report |
| `figures/.../raw_{label}{axis_tag}.png` | prepare | report |
| `figures/.../z-score_{col}.png` | plot | report |
| `figures/.../alert_{col}.png` | plot | - |
| `figures/.../{year}_{month}_summary.csv` | plot | report |
| `outputs/{opera_key}/{year}_{month}/` | report | - |
| `outputs/.../A4_{opera_key}_{year}_{month}.docx` | report | - |

## Summary CSV Columns

- `sensor_id`, `label`, `warnings`, `alarms`

## OPERE Mapping

P001→P001_Sommacampagna, P002→P002_Giuliari_Milani, P003→P003_Gua, P004→P004_Adige_Est, P005→P005_Adige_Ovest

## Label File Format

- `data/label-id/{OPERA_KEY}_label-id.csv`
- At least 2 columns: (label, id)
- id format: `BASE_suffix` (e.g. ICD_x), mapping uses `split("_")[0].upper()`
