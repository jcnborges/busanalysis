USE busanalysis_dw;

SELECT *
FROM vw_event
WHERE
	base_date = '2022-07-11'
    AND line_code = '303'
    AND itinerary_id = 1384
ORDER BY
	itinerary_id, vehicle, event_timestamp