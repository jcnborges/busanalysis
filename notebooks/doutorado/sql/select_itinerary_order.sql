USE busanalysis_dw;

SELECT DISTINCT *
FROM fat_itinerary
INNER JOIN dim_line ON dim_line.id = fat_itinerary.dim_line_id
INNER JOIN dim_bus_stop ON dim_bus_stop.id = fat_itinerary.bus_stop_id
WHERE 
	base_date = '2022-07-25'
    AND line_code = '020'
ORDER BY itinerary_id, seq

	