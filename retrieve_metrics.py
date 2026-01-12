import duckdb

con = duckdb.connect("metrics.duckdb")

# Preview all metrics
print(con.execute("SELECT * FROM pipeline_metrics ORDER BY ts DESC").fetchdf())

# Average CPU during the run
print(
    con.execute(
        "SELECT AVG(value) FROM pipeline_metrics WHERE name='cpu_mean'"
    ).fetchdf()
)

# Per-model execution times
print(
    con.execute(
        """
SELECT model, value AS execution_time
FROM pipeline_metrics
WHERE scope='model'
ORDER BY execution_time DESC
"""
    ).fetchdf()
)

con.close()
