USE busanalysis_dw;

SET @base_date = '2022-07-11'; -- Set the variable

WITH ItineraryCounts AS (
    SELECT DISTINCT line_code, base_date, COUNT(DISTINCT itinerary_id) AS itinerary_counts
	FROM fat_itinerary
	INNER JOIN dim_line ON dim_line.id = fat_itinerary.dim_line_id
	WHERE
		base_date = @base_date  -- Use the variable here
	GROUP BY line_code, base_date
	#HAVING COUNT(DISTINCT itinerary_id) <= 2
)
SELECT DISTINCT dim_line.line_code, fat_itinerary.itinerary_id, tbl.base_date
FROM fat_itinerary
INNER JOIN dim_line ON dim_line.id = fat_itinerary.dim_line_id
INNER JOIN ItineraryCounts tbl ON tbl.line_code = dim_line.line_code AND tbl.base_date = fat_itinerary.base_date
WHERE
	itinerary_counts <= 2;