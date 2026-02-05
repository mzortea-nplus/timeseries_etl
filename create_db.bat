cd dbt 
dbt run -s staging
dbt run -s all_static 
dbt run -s thermal_model