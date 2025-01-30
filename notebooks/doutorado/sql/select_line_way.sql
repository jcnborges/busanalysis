USE busanalysis_dw;

SELECT DISTINCT line_code, line_name, service_category, color, itinerary_id, line_way
FROM fat_itinerary
INNER JOIN dim_line ON dim_line.id = fat_itinerary.dim_line_id
WHERE 
	base_date = '2022-07-11'
    AND line_code = '050'

	