SET @base_date = '2022-07-17'; -- Set the variable

CREATE TABLE fat_most_relevant_itinerary AS
WITH ItineraryCounts AS (
    SELECT DISTINCT line_code, base_date, COUNT(DISTINCT itinerary_id) AS itinerary_counts
	FROM fat_itinerary
	INNER JOIN dim_line ON dim_line.id = fat_itinerary.dim_line_id
	WHERE
		base_date = @base_date  -- Use the variable here
	GROUP BY line_code, base_date
),
MostRelevantItineraries AS (
	SELECT dim_line.line_code, itinerary_id, COUNT(*) AS count, ROW_NUMBER() OVER (PARTITION BY dim_line.line_code ORDER BY COUNT(*) DESC) AS row_num, fat_event.base_date
	FROM fat_event
	INNER JOIN dim_line ON dim_line.id = fat_event.dim_line_id
	INNER JOIN ItineraryCounts tbl ON tbl.line_code = dim_line.line_code AND tbl.base_date = fat_event.base_date
    WHERE itinerary_counts > 2
	GROUP BY fat_event.base_date, dim_line.line_code, itinerary_id
)
SELECT line_code, itinerary_id, base_date
FROM MostRelevantItineraries
WHERE row_num = 1

UNION ALL

SELECT DISTINCT dim_line.line_code, fat_itinerary.itinerary_id, tbl.base_date
FROM fat_itinerary
INNER JOIN dim_line ON dim_line.id = fat_itinerary.dim_line_id
INNER JOIN ItineraryCounts tbl ON tbl.line_code = dim_line.line_code AND tbl.base_date = fat_itinerary.base_date
WHERE
	itinerary_counts <= 2;