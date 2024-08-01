USE busanalysis_dw;
SELECT 
	dim.name
    ,dim.latitude
    ,dim.longitude
    ,dim.type
    ,fat.line_code
    ,fat.event_timestamp
    ,fat.seq
    ,fat.vehicle
FROM fat_event AS fat
	INNER JOIN dim_bus_stop AS dim ON fat.dim_bus_stop_id = dim.id 
WHERE
	base_date = '2022-07-12'
    AND line_code = '829'
ORDER BY line_code, vehicle, event_timestamp