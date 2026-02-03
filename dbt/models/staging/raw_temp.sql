{{
config(
  materialized='table'
  )
}}

select 
    cast(time as timestamp) as time,
    * exclude(time) 
from read_parquet('{{ var("data_folder") }}/csv_temp/*.parquet')
where time between '{{ var("start_date") }}' and '{{ var("end_date") }}'