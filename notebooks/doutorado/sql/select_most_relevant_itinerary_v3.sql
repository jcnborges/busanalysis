WITH ItineraryCounts AS (
	SELECT DISTINCT dim_line_id, base_date, COUNT(DISTINCT itinerary_id) AS itinerary_counts
	FROM fat_itinerary
	WHERE
		base_date = '2022-07-11'
		AND dim_line_id = 614
	GROUP BY dim_line_id, base_date
),
MostRelevantItineraries AS (
	SELECT fat_event.dim_line_id, itinerary_id, COUNT(*) AS count, fat_event.base_date
	FROM fat_event
		INNER JOIN ItineraryCounts tbl ON tbl.dim_line_id = fat_event.dim_line_id AND tbl.base_date = fat_event.base_date
	WHERE itinerary_counts > 2
	GROUP BY fat_event.base_date, fat_event.dim_line_id, itinerary_id
),
ItineraryPairs AS (
SELECT 
	A.dim_line_id,    
	A.itinerary_id itinerary_id_A,
	A.count count_A, 
	B.itinerary_id itinerary_id_B,
	B.count count_B, 
	A.base_date
FROM MostRelevantItineraries A, MostRelevantItineraries B
WHERE
	A.dim_line_id = B.dim_line_id
	AND A.itinerary_id != B.itinerary_id
), 
EventsQuantity AS (
SELECT 
	ItineraryPairs.*,
	ROUND(
		COUNT(DISTINCT fat_event.dim_line_id, vehicle, dim_bus_stop_id, event_timestamp) /
		COUNT(DISTINCT fat_event.dim_line_id, vehicle, fat_event.itinerary_id, dim_bus_stop_id, event_timestamp),
		2
	) AS rate
FROM fat_event, ItineraryPairs
WHERE
	fat_event.dim_line_id = ItineraryPairs.dim_line_id
    AND fat_event.base_date = ItineraryPairs.base_date
	AND fat_event.itinerary_id IN (ItineraryPairs.itinerary_id_A, ItineraryPairs.itinerary_id_B)
GROUP BY 
	fat_event.dim_line_id, 
	ItineraryPairs.itinerary_id_A, 
	ItineraryPairs.count_A,
	ItineraryPairs.itinerary_id_B,
	ItineraryPairs.count_B,
	ItineraryPairs.base_date
),
BestCandidates AS
(
	SELECT 
		*, 
		ROW_NUMBER() OVER (PARTITION BY EventsQuantity.dim_line_id ORDER BY rate DESC, count_a DESC, count_b DESC) AS row_num
	FROM EventsQuantity
)
SELECT 
	line_code,
	itinerary_id, 
	BestCandidates.base_date
FROM BestCandidates
INNER JOIN MostRelevantItineraries ON 
	MostRelevantItineraries.dim_line_id = BestCandidates.dim_line_id
INNER JOIN dim_line ON dim_line.id = MostRelevantItineraries.dim_line_id    
WHERE
	BestCandidates.row_num = 1
	AND (
			(BestCandidates.rate >= 0.8 AND MostRelevantItineraries.itinerary_id IN (itinerary_id_A, itinerary_id_B))
			OR
			(BestCandidates.rate < 0.8 AND MostRelevantItineraries.itinerary_id = itinerary_id_A)
		)