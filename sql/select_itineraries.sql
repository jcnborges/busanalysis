USE busanalysis_dw;
SELECT 
	dim.name
    ,dim.latitude
    ,dim.longitude
    ,dim.type
    ,fat.line_code
    ,fat.itinerary_id
    ,fat.seq
FROM fat_itinerary AS fat
	INNER JOIN dim_bus_stop AS dim ON fat.bus_stop_id = dim.id 
WHERE
	base_date = '2022-07-11'