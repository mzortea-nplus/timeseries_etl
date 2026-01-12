import subprocess
import time
import psutil
import duckdb
import uuid
import json
import os
from datetime import datetime

# ---------------- CONFIG ----------------
DB_FILE = "metrics.duckdb"  # DuckDB file
WAREHOUSE_DIR = "../data/processed"  # folder containing your processed data
DBT_CMD = ["dbt", "run"]  # dbt command
RUN_RESULTS_JSON = "target/run_results.json"
# ---------------------------------------

RUN_ID = str(uuid.uuid4())
TIMESTAMP = datetime.utcnow()

# Connect to DuckDB
con = duckdb.connect(DB_FILE)

# Create metrics table if missing
con.execute(
    """
CREATE TABLE IF NOT EXISTS pipeline_metrics (
    ts        TIMESTAMP,
    scope     VARCHAR,      -- run | model | process | storage
    name      VARCHAR,      -- metric name
    value     DOUBLE,
    unit      VARCHAR,
    model     VARCHAR,
    extra     JSON
);
"""
)


def emit(scope, name, value, unit=None, model=None, extra=None):
    con.execute(
        "INSERT INTO pipeline_metrics VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.utcnow(),
            scope,
            name,
            float(value),
            unit,
            model,
            json.dumps(extra) if extra else None,
        ),
    )


# ---------------- RUN DBT ----------------
print(f"[{datetime.utcnow()}] Starting dbt run...")
start_time = time.time()
proc = subprocess.Popen(
    DBT_CMD, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
)

# Wait for dbt to finish
stdout, stderr = proc.communicate()
end_time = time.time()
runtime = end_time - start_time

# ---------------- CAPTURE CPU & MEMORY ----------------
cpu_samples = []
mem_samples = []

while True:
    if proc.poll() is not None:  # process finished
        break
    try:
        cpu_samples.append(p.cpu_percent(interval=0.5))  # half-second interval
        mem_samples.append(p.memory_info().rss)
    except psutil.NoSuchProcess:
        break

runtime = time.time() - start_time

# Compute averages
avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
max_mem = max(mem_samples) if mem_samples else 0

emit("process", "cpu_mean", avg_cpu, "percent")
emit("process", "rss_max", max_mem, "bytes")

print(f"Average CPU: {avg_cpu:.2f}%")
print(f"Peak memory: {max_mem/1024**2:.2f} MB")

# Save stdout/stderr for debugging
with open("dbt_stdout.log", "w") as f:
    f.write(stdout)
with open("dbt_stderr.log", "w") as f:
    f.write(stderr)

# ---------------- PER-MODEL EXECUTION TIMES ----------------
if os.path.exists(RUN_RESULTS_JSON):
    with open(RUN_RESULTS_JSON) as f:
        data = json.load(f)
    for r in data.get("results", []):
        emit(
            "model",
            "execution_time",
            r.get("execution_time", 0),
            "seconds",
            model=r.get("unique_id"),
            extra={"status": r.get("status")},
        )
else:
    print(f"Warning: {RUN_RESULTS_JSON} not found. Skipping per-model metrics.")

# ---------------- STORAGE USAGE ----------------
if os.path.exists(WAREHOUSE_DIR):
    size_bytes = sum(
        os.path.getsize(os.path.join(WAREHOUSE_DIR, f))
        for f in os.listdir(WAREHOUSE_DIR)
        if os.path.isfile(os.path.join(WAREHOUSE_DIR, f))
    )
    emit(
        "storage", "warehouse_size", size_bytes, "bytes", extra={"path": WAREHOUSE_DIR}
    )
    print(f"Warehouse size: {size_bytes / 1024**2:.2f} MB")
else:
    print(f"Warning: {WAREHOUSE_DIR} not found. Skipping storage metrics.")

con.close()
print(f"All metrics written to {DB_FILE}")
