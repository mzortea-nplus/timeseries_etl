cd dbt 
dbt run -s staging
dbt run -s all_static 
dbt run -s thermal_model
dbt run -s control
cd ..
python alert_dynamics.py
python report.py