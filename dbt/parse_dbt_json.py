import json
import duckdb
from datetime import datetime

DB_FILE = "../metrics.duckdb"

con = duckdb.connect(DB_FILE)

with open("target/run_results.json") as f:
    data = json.load(f)

for r in data["results"]:
    con.execute(
        "INSERT INTO pipeline_metrics VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.utcnow(),
            "model",
            "execution_time",
            r["execution_time"],
            "seconds",
            r["unique_id"],
            json.dumps({"status": r["status"]}),
        ),
    )

con.close()
