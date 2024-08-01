USE busanalysis_etl;
INSERT INTO busanalysis_dw.fat_event(line_code, vehicle, itinerary_id, status, event_timestamp, dim_bus_stop_id)
SELECT
	evt.line_code
    ,evt.vehicle
    ,evt.itinerary_id
    ,evt.status
    ,CAST(evt.event_timestamp AS DATETIME) AS event_timestamp
    ,dim.id AS dim_bus_stop_id
FROM etl_event AS evt
INNER JOIN etl_dim_bus_stop AS etl_dim ON evt.id = etl_dim.id
INNER JOIN busanalysis_dw.dim_bus_stop AS dim ON dim.name_norm = etl_dim.name_norm