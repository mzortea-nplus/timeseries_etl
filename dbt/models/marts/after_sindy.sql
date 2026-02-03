{{
    config(materialized='table')
}}

select 
    cast(time as timestamp) as time,
    * exclude(time) 
from read_parquet('C:/Users/m.zortea/Documents/timeseries_etl/dbt/*.parquet')