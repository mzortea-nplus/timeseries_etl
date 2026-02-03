{{
    config(materialized='table')
}}

WITH statics as  (
    SELECT 
        temp.time as temp_time, 
        spost.time as spost_time,
        est.time as est_time,
        incl.time as incl_time,
        temp.*  EXCLUDE (time),
        est.*   EXCLUDE (time),
        spost.* EXCLUDE (time),
        incl.*  EXCLUDE (time)
    FROM main_staging.raw_temp temp
    INNER JOIN main_staging.raw_est est
        ON date_trunc('second', temp.time) = date_trunc('second', est.time)
    INNER JOIN main_staging.raw_spost spost
        ON date_trunc('second', est.time) = date_trunc('second', spost.time)
    INNER JOIN main_staging.raw_incl incl
        ON date_trunc('second', spost.time) = date_trunc('second', incl.time)
)


SELECT
    temp_time as time,
    * EXCLUDE(spost_time, est_time, incl_time, temp_time),
    date_part('month', temp_time)  AS "month"
FROM statics
ORDER BY time