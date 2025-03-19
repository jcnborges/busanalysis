WITH ItineraryCounts AS (
	SELECT DISTINCT dim_line_id, base_date, COUNT(DISTINCT itinerary_id) AS itinerary_counts
	FROM fat_itinerary
	WHERE
		base_date = '2022-07-11'
	GROUP BY dim_line_id, base_date
),
MostRelevantItineraries AS (
	SELECT fat_event.dim_line_id, itinerary_id, COUNT(*) AS count, ROW_NUMBER() OVER (PARTITION BY fat_event.dim_line_id ORDER BY COUNT(*) DESC) AS row_num, fat_event.base_date
	FROM fat_event
	INNER JOIN ItineraryCounts tbl ON tbl.dim_line_id = fat_event.dim_line_id AND tbl.base_date = fat_event.base_date
	WHERE itinerary_counts > 2
	GROUP BY fat_event.base_date, fat_event.dim_line_id, itinerary_id
),
EventsQuantity AS (
	SELECT
		fat_event.dim_line_id,
		ROUND(
			COUNT(DISTINCT fat_event.dim_line_id, vehicle, dim_bus_stop_id, event_timestamp) /
			COUNT(DISTINCT fat_event.dim_line_id, vehicle, fat_event.itinerary_id, dim_bus_stop_id, event_timestamp),
			2
		) AS rate
	FROM fat_event
		INNER JOIN MostRelevantItineraries tbl ON tbl.dim_line_id = fat_event.dim_line_id
	WHERE
		fat_event.itinerary_id IN
		(
			SELECT itinerary_id 
			FROM MostRelevantItineraries
			WHERE
				dim_line_id = fat_event.dim_line_id
				AND row_num <= 2
		)
		AND row_num <= 2
	GROUP BY fat_event.dim_line_id
)
SELECT 
	line_code, itinerary_id, base_date
FROM MostRelevantItineraries
	INNER JOIN EventsQuantity ON EventsQuantity.dim_line_id = MostRelevantItineraries.dim_line_id
    INNER JOIN dim_line ON dim_line.id = MostRelevantItineraries.dim_line_id
WHERE
	(row_num < 2 and rate < 0.8) or (row_num <= 2 and rate >= 0.8)
	
	